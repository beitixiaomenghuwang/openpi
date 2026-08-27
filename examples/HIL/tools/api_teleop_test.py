#!/usr/bin/env python3
"""Standalone API endpoint-mode VR teleoperation test.

This process is independent of the HIL action-chunk/supervisor path. It does
not import or call the HIL supervisor, policy server, router, or action-chunk
contract. It subscribes to the second S100's Zenoh
``sec/xr/devicepose`` stream, maps the left/right controller poses into the
robot convention, and publishes the robot API endpoint topics directly when
``--enable-api-output`` is supplied.

The start sequence is intentionally explicit:

1. Right A requests teleoperation (or ``--auto-start`` starts after readiness).
2. The script sends ``/api/fsm/enable = 0`` and waits for robot ``PAUSE``.
3. It snapshots the latest robot endpoint poses and the latest mapped controller
   poses as a takeover anchor.
4. It enables the robot and waits for ``READY`` before applying relative hand
   motion. Therefore the first teleop command equals the current robot pose.

Right A starts teleoperation and Right B stops it. On stop, the script holds
the latest actual endpoint poses, disables the API FSM, and reports lock
confirmation only after the robot returns ``FSM_PAUSE``. The terminal reports
VR app connection, start, and stop/lock events.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np

from hil import topics
from hil.pose_math import (
    anchored_pose,
    apply_tool_rotation,
    as_vector,
    change_basis,
    clamp_arm_reach,
    limit_pose_step,
    normalize_quaternion,
    quaternion_to_matrix,
)
from hil.sec_dev_protocol import DecodedSecDevPose as DecodedPose
from hil.sec_dev_protocol import (
    PoseValue,
    decode_sec_dev_pose,
    resolve_controller_analog,
)
from hil.zenoh_pose_sampler import WireDecodeError


LOGGER = logging.getLogger("api_teleop_test")

FSM_ERROR = -1
FSM_PAUSE = 0
FSM_READY = 2

# Same calibrated WebXR-to-robot transform used by the existing teleop server.
WEBXR_TO_ROBOT_BASIS = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
LEFT_TOOL_ROTATION = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
RIGHT_TOOL_ROTATION = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

RIGHT_A = 0x04
RIGHT_B = 0x08
LEFT_Y = 0x02


@dataclass
class Timed:
    value: Any = None
    received_at: float = 0.0

    def age(self, now: float) -> float:
        if self.value is None:
            return math.inf
        return max(0.0, now - self.received_at)


def _payload_bytes(payload: Any) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if hasattr(payload, "to_bytes"):
        return payload.to_bytes()
    return bytes(payload)


class DirectAPITeleop:
    def __init__(self, args):
        import rclpy
        from geometry_msgs.msg import Pose
        from std_msgs.msg import Float32, Int32, String

        self.rclpy = rclpy
        self.args = args
        self.Pose = Pose
        self.Float32 = Float32
        self.lock = threading.RLock()
        self.node = rclpy.create_node("api_direct_vr_teleop_test")
        self.output_enabled = bool(args.enable_api_output)

        self.phase = "paused"
        self.phase_started_at = time.monotonic()
        self.last_reason = "startup; waiting for Right A"
        self.start_requested = bool(args.auto_start)
        self.pause_requested = False
        self.pause_reason = ""
        self.previous_mask: int | None = None
        self.vr_connected = False
        self.lock_confirmation_pending = False
        self.lock_requested_at = 0.0
        self.lock_timeout_reported = False
        self.latest_pose = Timed()
        self.robot_poses = {"left": Timed(), "right": Timed()}
        self.fsm_state = Timed()
        self.current_mode = Timed()
        self.last_decode_error = ""
        self.packets_received = 0
        self.packets_rejected = 0
        self._ownership_ok = False
        self._last_disable_at = 0.0
        self._last_log = 0.0

        self.raw_grippers = {"left": 0.10, "right": 0.10}
        self.last_grippers = {"left": 0.10, "right": 0.10}
        self.raw_anchors: dict[str, PoseValue] = {}
        self.robot_anchors: dict[str, PoseValue] = {}
        self.last_outputs: dict[str, PoseValue | None] = {"left": None, "right": None}

        self.pose_publishers = {}
        self.gripper_publishers = {}
        self.enable_publisher = None
        if self.output_enabled:
            self.pose_publishers = {
                "left": self.node.create_publisher(Pose, topics.API_LEFT_POSE, 10),
                "right": self.node.create_publisher(Pose, topics.API_RIGHT_POSE, 10),
            }
            self.gripper_publishers = {
                "left": self.node.create_publisher(Float32, topics.API_LEFT_GRIPPER, 10),
                "right": self.node.create_publisher(Float32, topics.API_RIGHT_GRIPPER, 10),
            }
            self.enable_publisher = self.node.create_publisher(
                Float32, topics.API_FSM_ENABLE, 10
            )

        self.node.create_subscription(
            Pose,
            topics.ROBOT_LEFT_POSE,
            lambda msg: self._on_robot_pose("left", msg),
            10,
        )
        self.node.create_subscription(
            Pose,
            topics.ROBOT_RIGHT_POSE,
            lambda msg: self._on_robot_pose("right", msg),
            10,
        )
        self.node.create_subscription(Int32, topics.ROBOT_FSM_STATE, self._on_fsm, 10)
        self.node.create_subscription(String, topics.ROBOT_CURRENT_MODE, self._on_mode, 10)
        self.node.create_timer(1.0 / args.control_hz, self._control_tick)
        self.node.create_timer(1.0 / args.heartbeat_hz, self._heartbeat_tick)
        self.node.create_timer(0.5, self._ownership_tick)
        self.node.create_timer(2.0, self._status_tick)

        self._publish_enable(0.0)
        LOGGER.info(
            "[STANDALONE API TELEOP] HIL supervisor/router/policy are not used; "
            "output=%s; waiting for VR app pose stream",
            "ENABLED" if self.output_enabled else "MONITOR ONLY",
        )

    @staticmethod
    def _pose_from_msg(msg) -> PoseValue:
        position = as_vector((msg.position.x, msg.position.y, msg.position.z), 3)
        quaternion = normalize_quaternion(
            (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        )
        return PoseValue(position=position, quaternion=quaternion)

    def _make_pose_msg(self, value: PoseValue):
        msg = self.Pose()
        msg.position.x, msg.position.y, msg.position.z = map(float, value.position)
        (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ) = map(float, value.quaternion)
        return msg

    def _on_robot_pose(self, side: str, msg) -> None:
        try:
            value = self._pose_from_msg(msg)
        except ValueError as exc:
            LOGGER.error("Invalid %s current endpoint pose: %s", side, exc)
            return
        with self.lock:
            self.robot_poses[side] = Timed(value, time.monotonic())

    def _on_fsm(self, msg) -> None:
        value = int(msg.data)
        with self.lock:
            self.fsm_state = Timed(value, time.monotonic())
            if value == FSM_PAUSE and self.lock_confirmation_pending:
                self.lock_confirmation_pending = False
                self.lock_timeout_reported = False
                self.last_reason = "teleoperation stopped; FSM PAUSE confirmed"
                LOGGER.warning(
                    "[ROBOT LOCKED] /fsm_state=PAUSE confirmed; press Right A to restart"
                )

    def _on_mode(self, msg) -> None:
        try:
            value = json.loads(msg.data)
            if not isinstance(value, dict):
                raise ValueError("mode message is not an object")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.error("Invalid /api/current_mode: %s", exc)
            return
        with self.lock:
            self.current_mode = Timed(value, time.monotonic())

    def on_zenoh_sample(self, sample) -> None:
        try:
            packet = decode_sec_dev_pose(_payload_bytes(sample.payload))
        except (TypeError, ValueError, WireDecodeError) as exc:
            with self.lock:
                self.packets_rejected += 1
                self.last_decode_error = str(exc)
            return

        now = time.monotonic()
        with self.lock:
            if not self.vr_connected:
                self.vr_connected = True
                LOGGER.info(
                    "[VR CONNECTED] controller pose stream is live; "
                    "press Right A to start, Right B to stop and lock"
                )
            self.latest_pose = Timed(packet, now)
            self.packets_received += 1
            for side in ("left", "right"):
                self.raw_grippers[side] = resolve_controller_analog(
                    packet,
                    side,
                    self.args.gripper_input,
                    self.raw_grippers[side],
                    released_value=0.0,
                )

            previous = self.previous_mask
            self.previous_mask = packet.input_mask
            if previous is None:
                return
            rising = packet.input_mask & ~previous
            if rising & RIGHT_B or rising & LEFT_Y:
                self.pause_requested = True
                self.pause_reason = (
                    "Right B stop requested" if rising & RIGHT_B else "Left Y stop requested"
                )
                LOGGER.warning(
                    "[TELEOP STOP REQUEST] %s; holding endpoint poses and disabling API FSM",
                    "Right B" if rising & RIGHT_B else "Left Y",
                )
            elif rising & RIGHT_A:
                if self.phase == "paused":
                    self.start_requested = True
                    LOGGER.info("[TELEOP START REQUEST] Right A received; preparing takeover anchor")
                else:
                    LOGGER.info("[RIGHT A] ignored because teleoperation is already starting/active")

    def _mode_valid(self, now: float) -> tuple[bool, str]:
        if self.args.skip_mode_check:
            return True, "mode check disabled"
        if self.current_mode.age(now) > self.args.mode_timeout:
            return False, "missing or stale /api/current_mode"
        mode = self.current_mode.value
        checks = (
            (mode.get("meta_mode") == 1, "robot is not in API mode"),
            (mode.get("left_arm_control_mode") == 0, "left arm is not endpoint mode"),
            (mode.get("right_arm_control_mode") == 0, "right arm is not endpoint mode"),
            (bool(mode.get("enable_left_arm")), "left arm is disabled"),
            (bool(mode.get("enable_right_arm")), "right arm is disabled"),
        )
        for passed, reason in checks:
            if not passed:
                return False, reason
        return True, "ok"

    def _feedback_valid(self, now: float) -> tuple[bool, str]:
        for side in ("left", "right"):
            if self.robot_poses[side].age(now) > self.args.feedback_timeout:
                return False, f"missing or stale {side} current_ee_pose"
        if self.fsm_state.age(now) > self.args.feedback_timeout:
            return False, "missing or stale /fsm_state"
        if self.fsm_state.value == FSM_ERROR:
            return False, "robot FSM is ERROR"
        return True, "ok"

    def _pose_valid(self, now: float) -> tuple[bool, str]:
        if self.latest_pose.age(now) > self.args.pose_timeout:
            if self.vr_connected:
                self.vr_connected = False
                LOGGER.warning(
                    "[VR DISCONNECTED] controller pose stream is stale; stopping teleoperation"
                )
            return False, "missing or stale sec_dev controller pose"
        return True, "ok"

    def _ownership_tick(self) -> None:
        now = time.monotonic()
        if not self.output_enabled:
            with self.lock:
                self._ownership_ok = True
            return
        if self.args.skip_ownership_check:
            with self.lock:
                self._ownership_ok = True
            return
        conflicts = []
        topics_to_check = (
            topics.API_LEFT_POSE,
            topics.API_RIGHT_POSE,
            topics.API_LEFT_GRIPPER,
            topics.API_RIGHT_GRIPPER,
            topics.API_FSM_ENABLE,
        )
        own_name = self.node.get_name()
        own_namespace = self.node.get_namespace()
        for topic_name in topics_to_check:
            infos = self.node.get_publishers_info_by_topic(topic_name)
            own = [
                info
                for info in infos
                if info.node_name == own_name and info.node_namespace == own_namespace
            ]
            if len(infos) != 1 or len(own) != 1:
                names = sorted(
                    f"{info.node_namespace.rstrip('/')}/{info.node_name}" for info in infos
                )
                conflicts.append(f"{topic_name} publishers={names}")
        valid = not conflicts
        with self.lock:
            changed = valid != self._ownership_ok
            self._ownership_ok = valid
        if changed or not valid and now - self._last_log > 2.0:
            self._last_log = now
            if valid:
                LOGGER.info("Direct API publisher ownership check passed")
            else:
                LOGGER.warning("API ownership check failed: %s", "; ".join(conflicts))

    def _publish_enable(self, value: float) -> None:
        if not self.output_enabled:
            return
        self.enable_publisher.publish(self.Float32(data=float(value)))
        if value == 0.0:
            self._last_disable_at = time.monotonic()

    def _pause_locked(self, reason: str) -> None:
        was_running = self.phase != "paused"
        changed = self.phase != "paused" or self.last_reason != reason
        self._hold_current_robot_locked()
        self.phase = "paused"
        self.start_requested = False
        self.pause_requested = False
        self.pause_reason = ""
        self.raw_anchors.clear()
        self.robot_anchors.clear()
        self.last_outputs = {"left": None, "right": None}
        self.last_reason = reason
        self._publish_enable(0.0)
        if changed:
            if self.output_enabled and was_running and self.fsm_state.value != FSM_PAUSE:
                self.lock_confirmation_pending = True
                self.lock_requested_at = time.monotonic()
                self.lock_timeout_reported = False
                LOGGER.warning(
                    "[TELEOP STOPPED] %s; waiting for /fsm_state=PAUSE lock confirmation",
                    reason,
                )
            elif self.output_enabled and self.fsm_state.value == FSM_PAUSE:
                self.lock_confirmation_pending = False
                LOGGER.warning("[TELEOP STOPPED / ROBOT LOCKED] %s", reason)
            elif not self.output_enabled:
                LOGGER.warning("[TELEOP STOPPED] %s (monitor-only, no robot command sent)", reason)
            else:
                LOGGER.warning("[TELEOP STOPPED] %s", reason)

    def _hold_current_robot_locked(self) -> None:
        """Send the latest actual endpoint pose before disabling the API FSM."""
        now = time.monotonic()
        for side in ("left", "right"):
            current = self.robot_poses[side]
            if current.value is not None and current.age(now) <= self.args.feedback_timeout:
                value = current.value
                self.last_outputs[side] = PoseValue(
                    value.position.copy(), value.quaternion.copy()
                )
        self._publish_hold_locked()

    def _ready_for_start_locked(self, now: float) -> tuple[bool, str]:
        for check in (
            self._mode_valid(now),
            self._feedback_valid(now),
            self._pose_valid(now),
        ):
            if not check[0]:
                return check
        if not self._ownership_ok and not self.args.skip_ownership_check:
            return False, "API publisher ownership is not exclusive"
        return True, "ok"

    def _map_controller_pose(self, side: str, raw: PoseValue) -> PoseValue:
        position, quaternion = change_basis(
            raw.position, raw.quaternion, WEBXR_TO_ROBOT_BASIS
        )
        wrist_offset = self.args.wrist_offsets[side]
        position = position - quaternion_to_matrix(quaternion) @ wrist_offset
        tool_rotation = LEFT_TOOL_ROTATION if side == "left" else RIGHT_TOOL_ROTATION
        quaternion = apply_tool_rotation(quaternion, tool_rotation)
        return PoseValue(position=position, quaternion=quaternion)

    def _begin_start_locked(self, now: float) -> None:
        packet = self.latest_pose.value
        if packet is None:
            self._pause_locked("cannot start without a current controller pose")
            return
        for side in ("left", "right"):
            robot = self.robot_poses[side].value
            mapped = self._map_controller_pose(side, packet.poses[side])
            self.robot_anchors[side] = PoseValue(
                robot.position.copy(), robot.quaternion.copy()
            )
            self.raw_anchors[side] = PoseValue(
                mapped.position.copy(), mapped.quaternion.copy()
            )
            self.last_outputs[side] = PoseValue(
                robot.position.copy(), robot.quaternion.copy()
            )
        self.phase = "starting"
        self.phase_started_at = now
        self.start_requested = False
        self.last_reason = "enabled; waiting for FSM READY"
        self._publish_hold_locked()
        self._publish_enable(1.0)
        LOGGER.info(
            "[TAKEOVER ANCHOR CAPTURED] first command equals current robot pose; "
            "waiting for FSM READY"
        )

    def _start_request_locked(self, now: float) -> None:
        valid, reason = self._ready_for_start_locked(now)
        if not valid:
            self.start_requested = False
            self.last_reason = f"start rejected: {reason}"
            LOGGER.warning(self.last_reason)
            return
        self.phase = "waiting_for_pause"
        self.phase_started_at = now
        self.last_reason = "waiting for FSM PAUSE before anchoring"
        self._publish_enable(0.0)
        LOGGER.info("[TELEOP STARTING] disabling API FSM and waiting for FSM PAUSE")

    def _publish_hold_locked(self) -> None:
        for side in ("left", "right"):
            output = self.last_outputs[side]
            if output is None:
                continue
            self._publish_pose_and_gripper(side, output, self.last_grippers[side])

    def _publish_pose_and_gripper(self, side: str, pose: PoseValue, gripper: float) -> None:
        if not self.output_enabled:
            return
        self.pose_publishers[side].publish(self._make_pose_msg(pose))
        self.gripper_publishers[side].publish(
            self.Float32(data=float(np.clip(gripper, 0.0, 1.0)))
        )

    def _publish_active_locked(self, now: float) -> None:
        packet: DecodedPose = self.latest_pose.value
        if packet is None:
            self._pause_locked("controller pose disappeared")
            return
        dt = 1.0 / self.args.control_hz
        for side in ("left", "right"):
            raw = self._map_controller_pose(side, packet.poses[side])
            raw_anchor = self.raw_anchors[side]
            robot_anchor = self.robot_anchors[side]
            position, quaternion = anchored_pose(
                raw.position,
                raw.quaternion,
                raw_anchor.position,
                raw_anchor.quaternion,
                robot_anchor.position,
                robot_anchor.quaternion,
                position_scale=self.args.position_scale,
            )
            displacement = position - robot_anchor.position
            distance = float(np.linalg.norm(displacement))
            if distance > self.args.max_teleop_displacement:
                position = robot_anchor.position + displacement * (
                    self.args.max_teleop_displacement / distance
                )
            position = clamp_arm_reach(position, self.args.max_arm_reach)

            previous = self.last_outputs[side]
            if previous is None:
                previous = robot_anchor
            position, quaternion = limit_pose_step(
                previous.position,
                previous.quaternion,
                position,
                quaternion,
                self.args.max_translation_speed * dt,
                self.args.max_rotation_speed * dt,
            )
            output = PoseValue(position=position, quaternion=quaternion)
            self.last_outputs[side] = output

            desired = self.raw_grippers[side]
            rate = (
                self.args.max_gripper_release_rate
                if desired < self.last_grippers[side]
                else self.args.max_gripper_rate
            )
            max_step = rate * dt
            next_gripper = self.last_grippers[side] + float(
                np.clip(desired - self.last_grippers[side], -max_step, max_step)
            )
            self.last_grippers[side] = float(np.clip(next_gripper, 0.0, 1.0))
            self._publish_pose_and_gripper(side, output, self.last_grippers[side])

    def _check_active_locked(self, now: float) -> bool:
        checks = (
            self._mode_valid(now),
            self._feedback_valid(now),
            self._pose_valid(now),
        )
        for valid, reason in checks:
            if not valid:
                self._pause_locked(reason)
                return False
        if not self._ownership_ok and not self.args.skip_ownership_check:
            self._pause_locked("API publisher ownership lost")
            return False
        # During the startup handshake the robot is expected to remain in
        # PAUSE/SLOW_START until it reaches READY. Only an active teleop loop
        # treats PAUSE as an unexpected loss of ownership.
        if self.phase == "active" and self.fsm_state.value in (FSM_PAUSE, FSM_ERROR):
            self._pause_locked(f"robot FSM entered {self.fsm_state.value}")
            return False
        return True

    def _control_tick(self) -> None:
        now = time.monotonic()
        with self.lock:
            if self.pause_requested:
                self._pause_locked(self.pause_reason or "controller stop requested")
                return
            if self.phase == "paused":
                if self.start_requested:
                    self._start_request_locked(now)
                return

            if self.phase == "waiting_for_pause":
                valid, reason = self._ready_for_start_locked(now)
                if not valid:
                    self._pause_locked(f"start became unsafe: {reason}")
                    return
                if now - self._last_disable_at >= 0.1:
                    self._publish_enable(0.0)
                if self.fsm_state.value == FSM_PAUSE and now - self.phase_started_at >= 0.1:
                    self._begin_start_locked(now)
                elif now - self.phase_started_at > self.args.pause_timeout:
                    self._pause_locked("timed out waiting for FSM PAUSE")
                return

            if not self._check_active_locked(now):
                return
            if self.phase == "starting":
                self._publish_hold_locked()
                self._publish_enable(1.0)
                if self.fsm_state.value == FSM_READY:
                    self.phase = "active"
                    self.phase_started_at = now
                    self.last_reason = "teleop active"
                    LOGGER.info(
                        "[TELEOP ACTIVE] endpoint commands are being published; "
                        "press Right B to stop"
                    )
                elif now - self.phase_started_at > self.args.ready_timeout:
                    self._pause_locked("timed out waiting for FSM READY")
                return

            if self.phase == "active":
                self._publish_active_locked(now)
                self._publish_enable(1.0)

    def _heartbeat_tick(self) -> None:
        with self.lock:
            if self.phase in ("waiting_for_pause", "starting", "active"):
                self._publish_enable(1.0 if self.phase != "waiting_for_pause" else 0.0)
            else:
                self._publish_enable(0.0)

    def _status_tick(self) -> None:
        now = time.monotonic()
        with self.lock:
            if (
                self.lock_confirmation_pending
                and not self.lock_timeout_reported
                and now - self.lock_requested_at > self.args.pause_timeout
            ):
                self.lock_timeout_reported = True
                LOGGER.error(
                    "[LOCK NOT CONFIRMED] /api/fsm/enable remains 0 but FSM PAUSE "
                    "was not received within %.1fs",
                    self.args.pause_timeout,
                )
            LOGGER.debug(
                "status phase=%s vr_connected=%s fsm=%s pose_age=%.3fs packets=%d rejected=%d reason=%s",
                self.phase,
                self.vr_connected,
                self.fsm_state.value,
                self.latest_pose.age(now),
                self.packets_received,
                self.packets_rejected,
                self.last_reason,
            )

    def close(self) -> None:
        with self.lock:
            self._pause_locked("shutdown")
            if self.output_enabled:
                for _ in range(3):
                    self.enable_publisher.publish(self.Float32(data=0.0))
        self.node.destroy_node()


def parse_offset(text: str) -> np.ndarray:
    values = np.asarray([float(value) for value in text.split(",")], dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("Expected three finite comma-separated values")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default=topics.VR_ZENOH_DEVICEPOSE_KEY)
    parser.add_argument(
        "--zenoh-endpoint",
        "--endpoint",
        dest="zenoh_endpoint",
        default="",
        help="Optional Zenoh connect endpoint; omit for automatic discovery",
    )
    parser.add_argument("--control-hz", type=float, default=60.0)
    parser.add_argument("--heartbeat-hz", type=float, default=20.0)
    parser.add_argument("--pose-timeout", type=float, default=0.30)
    parser.add_argument("--feedback-timeout", type=float, default=0.50)
    parser.add_argument("--mode-timeout", type=float, default=3.0)
    parser.add_argument("--pause-timeout", type=float, default=2.0)
    parser.add_argument("--ready-timeout", type=float, default=5.0)
    parser.add_argument("--max-arm-reach", type=float, default=0.55)
    parser.add_argument("--max-teleop-displacement", type=float, default=0.40)
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--max-translation-speed", type=float, default=0.25)
    parser.add_argument("--max-rotation-speed", type=float, default=1.20)
    parser.add_argument("--max-gripper-rate", type=float, default=2.0)
    parser.add_argument("--max-gripper-release-rate", type=float, default=10.0)
    parser.add_argument("--gripper-input", choices=("trigger", "squeeze"), default="trigger")
    parser.add_argument("--left-wrist-offset", type=parse_offset, default=np.array([0.13205294, -0.03416498, 0.03397751]))
    parser.add_argument("--right-wrist-offset", type=parse_offset, default=np.array([0.12516124, 0.04039848, 0.03931866]))
    parser.add_argument("--auto-start", action="store_true", help="Start after feedback/pose readiness without Right A")
    parser.add_argument("--enable-api-output", action="store_true", help="Actually publish /api/* commands; otherwise monitor only")
    parser.add_argument("--skip-mode-check", action="store_true")
    parser.add_argument("--skip-ownership-check", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.control_hz <= 0.0 or args.heartbeat_hz <= 0.0:
        parser.error("control/heartbeat rates must be positive")
    if args.pose_timeout <= 0.0 or args.feedback_timeout <= 0.0:
        parser.error("pose/feedback timeouts must be positive")
    if args.position_scale <= 0.0:
        parser.error("--position-scale must be positive")
    if args.max_gripper_rate <= 0.0 or args.max_gripper_release_rate <= 0.0:
        parser.error("gripper rates must be positive")
    args.wrist_offsets = {
        "left": np.asarray(args.left_wrist_offset, dtype=np.float64),
        "right": np.asarray(args.right_wrist_offset, dtype=np.float64),
    }
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    try:
        import zenoh
        import rclpy
    except ImportError as exc:
        LOGGER.error("Missing runtime dependency: %s", exc)
        return 2

    rclpy.init()
    node = DirectAPITeleop(args)
    config = zenoh.Config()
    if args.zenoh_endpoint:
        config.insert_json5("connect/endpoints", json.dumps([args.zenoh_endpoint]))
    session = zenoh.open(config)
    subscriber = session.declare_subscriber(args.key, node.on_zenoh_sample)
    LOGGER.info(
        "Subscribed to %s via %s; %s",
        args.key,
        args.zenoh_endpoint or "Zenoh auto-discovery",
        "API output ENABLED" if args.enable_api_output else "monitor-only dry run",
    )
    try:
        rclpy.spin(node.node)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted")
    finally:
        try:
            subscriber.undeclare()
        except Exception as exc:
            LOGGER.warning("Zenoh subscriber cleanup failed: %s", exc)
        try:
            session.close()
        except Exception as exc:
            LOGGER.warning("Zenoh session cleanup failed: %s", exc)
        node.close()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
