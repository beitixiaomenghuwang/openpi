#!/usr/bin/env python3
"""Single owner and safety supervisor for API endpoint-mode HIL control."""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from . import topics
    from .action_chunk import ActionChunk, ActionFrame, decode_action_chunk
    from .pose_math import (
        anchored_pose,
        as_vector,
        clamp_arm_reach,
        limit_pose_step,
        normalize_quaternion,
        quaternion_angle,
    )
except ImportError:
    import topics
    from action_chunk import ActionChunk, ActionFrame, decode_action_chunk
    from pose_math import (
        anchored_pose,
        as_vector,
        clamp_arm_reach,
        limit_pose_step,
        normalize_quaternion,
        quaternion_angle,
    )


LOGGER = logging.getLogger("hil_supervisor")
FSM_ERROR = -1
FSM_PAUSE = 0
FSM_SLOW_START = 1
FSM_READY = 2


@dataclass
class TimedValue:
    value: object = None
    received_at: float = 0.0

    def age(self, now: float) -> float:
        return math.inf if self.value is None else max(0.0, now - self.received_at)


@dataclass
class SourceState:
    latest_chunk: Optional[ActionChunk] = None
    chunk_received_at: float = 0.0
    latest_created_at_ns: int = 0
    latest_sequence: int = -1
    latest_action: Optional[ActionFrame] = None
    invalid_reason: str = ""
    superseded_actions: int = 0
    accepted_chunks: int = 0
    rejected_chunks: int = 0
    last_rejection_reason: str = ""


@dataclass
class PoseArray:
    position: np.ndarray
    quaternion: np.ndarray


def policy_handover_target_reached(
    action: ActionFrame,
    robot_poses: dict[str, PoseArray],
    *,
    position_limit: float,
    rotation_limit: float,
) -> tuple[bool, str]:
    """Report whether both endpoints are close to the latest policy target."""

    max_translation = 0.0
    max_rotation = 0.0
    for side in ("left", "right"):
        command = action.command(side)
        robot = robot_poses[side]
        translation = float(np.linalg.norm(command.position - robot.position))
        rotation = quaternion_angle(command.quaternion, robot.quaternion)
        max_translation = max(max_translation, translation)
        max_rotation = max(max_rotation, rotation)
    reached = (
        max_translation <= position_limit and max_rotation <= rotation_limit
    )
    return reached, (
        f"latest target error {max_translation:.3f} m, {max_rotation:.3f} rad; "
        f"tolerances are {position_limit:.3f} m, {rotation_limit:.3f} rad"
    )


class HILSupervisor:
    def __init__(self, args):
        import rclpy
        from geometry_msgs.msg import Pose
        from rclpy.qos import QoSProfile
        from std_msgs.msg import Float32, Int32, String

        self.args = args
        self.rclpy = rclpy
        self.Pose = Pose
        self.Float32 = Float32
        self.String = String
        self.node = rclpy.create_node("hil_api_endpoint_supervisor")
        teleop_qos = QoSProfile(depth=1)
        self.lock = threading.RLock()
        self.sources = {"policy": SourceState(), "teleop": SourceState()}
        self.robot_poses = {"left": TimedValue(), "right": TimedValue()}
        self.fsm_state = TimedValue()
        self.current_mode = TimedValue()
        self.api_ownership = TimedValue(
            (False, "API ownership has not been checked"), 0.0
        )
        self.teleop_server_status = TimedValue()
        self.policy_request_received_at = 0.0
        self.phase = "paused"
        self.selected_mode = "pause"
        self.pending_mode: Optional[str] = None
        self.phase_started_at = time.monotonic()
        self.last_reason = "startup"
        self.last_outputs: dict[str, Optional[PoseArray]] = {"left": None, "right": None}
        # TA2 documentation identifies 0.10 as approximately zero feedforward.
        self.last_grippers = {"left": 0.10, "right": 0.10}
        self.teleop_raw_anchors: dict[str, PoseArray] = {}
        self.teleop_robot_anchors: dict[str, PoseArray] = {}
        self._last_disable_at = 0.0

        self.api_pose_publishers = {
            "left": self.node.create_publisher(Pose, topics.API_LEFT_POSE, 10),
            "right": self.node.create_publisher(Pose, topics.API_RIGHT_POSE, 10),
        }
        self.api_gripper_publishers = {
            "left": self.node.create_publisher(Float32, topics.API_LEFT_GRIPPER, 10),
            "right": self.node.create_publisher(Float32, topics.API_RIGHT_GRIPPER, 10),
        }
        self.enable_publisher = self.node.create_publisher(Float32, topics.API_FSM_ENABLE, 10)
        self.status_publisher = self.node.create_publisher(String, topics.MODE_STATUS, 10)

        self.node.create_subscription(String, topics.MODE_REQUEST, self._on_mode_request, 10)
        self.node.create_subscription(Pose, topics.ROBOT_LEFT_POSE, lambda msg: self._on_robot_pose("left", msg), 10)
        self.node.create_subscription(Pose, topics.ROBOT_RIGHT_POSE, lambda msg: self._on_robot_pose("right", msg), 10)
        self.node.create_subscription(Int32, topics.ROBOT_FSM_STATE, self._on_fsm_state, 10)
        self.node.create_subscription(String, topics.ROBOT_CURRENT_MODE, self._on_current_mode, 10)
        self.node.create_subscription(
            String,
            topics.TELEOP_SERVER_STATUS,
            self._on_teleop_server_status,
            10,
        )

        for source in ("policy", "teleop"):
            self.node.create_subscription(
                String,
                topics.action_chunk_topic(source),
                lambda msg, source=source: self._on_source_chunk(source, msg),
                teleop_qos if source == "teleop" else 10,
            )

        self.node.create_timer(1.0 / args.control_hz, self._control_tick)
        self.node.create_timer(1.0 / args.heartbeat_hz, self._heartbeat_tick)
        self.node.create_timer(0.5, self._ownership_tick)
        self.node.create_timer(0.2, self._status_tick)
        self._publish_enable_locked(0.0)
        LOGGER.info(
            "Supervisor started in PAUSE; endpoint API ownership is exclusive, dry_run=%s",
            args.dry_run,
        )

    @staticmethod
    def _pose_to_array(pose) -> PoseArray:
        position = as_vector((pose.position.x, pose.position.y, pose.position.z), 3)
        quaternion = normalize_quaternion(
            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        )
        return PoseArray(position=position, quaternion=quaternion)

    def _array_to_pose(self, value: PoseArray):
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
            value = self._pose_to_array(msg)
        except ValueError as exc:
            LOGGER.error("Rejected invalid %s robot endpoint pose: %s", side, exc)
            return
        with self.lock:
            self.robot_poses[side] = TimedValue(value, time.monotonic())

    def _on_fsm_state(self, msg) -> None:
        now = time.monotonic()
        with self.lock:
            self.fsm_state = TimedValue(int(msg.data), now)
            if int(msg.data) == FSM_ERROR and self.phase != "paused":
                self._pause_locked("robot FSM entered ERROR", now)

    def _on_current_mode(self, msg) -> None:
        try:
            value = json.loads(msg.data)
            if not isinstance(value, dict):
                raise ValueError("JSON root is not an object")
        except (json.JSONDecodeError, ValueError) as exc:
            LOGGER.error("Rejected malformed /api/current_mode: %s", exc)
            return
        now = time.monotonic()
        with self.lock:
            self.current_mode = TimedValue(value, now)
            valid, reason = self._mode_configuration_valid(now)
            if not valid and self.phase != "paused":
                self._pause_locked(reason, now)

    def _on_teleop_server_status(self, msg) -> None:
        try:
            value = json.loads(msg.data)
            if not isinstance(value, dict):
                raise ValueError("JSON root is not an object")
        except (json.JSONDecodeError, ValueError) as exc:
            LOGGER.error("Rejected malformed teleop server status: %s", exc)
            return
        with self.lock:
            self.teleop_server_status = TimedValue(value, time.monotonic())

    def _reject_source_chunk_locked(
        self, source: str, reason: str, now: float, *, blocking: bool
    ) -> None:
        state = self.sources[source]
        reason_changed = reason != state.last_rejection_reason
        state.rejected_chunks += 1
        state.last_rejection_reason = reason
        if blocking:
            state.invalid_reason = reason
            if self.selected_mode == source:
                self._pause_locked(reason, now)
        if reason_changed:
            waiting = (
                getattr(self, "pending_mode", None) == source
                and self.selected_mode == "pause"
            )
            suffix = "；机器人保持锁停，等待下一份动作" if waiting else ""
            LOGGER.warning("[%s] 拒绝动作 chunk：%s%s", source.upper(), reason, suffix)

    def _on_source_chunk(self, source: str, msg) -> None:
        now = time.monotonic()
        try:
            chunk = decode_action_chunk(
                msg.data,
                expected_source=source,
                expected_frame_ids=topics.ACTION_FRAME_IDS[source],
                max_actions=self.args.max_chunk_actions,
            )
        except ValueError as exc:
            with self.lock:
                self._reject_source_chunk_locked(
                    source, f"invalid {source} chunk: {exc}", now, blocking=True
                )
            return

        with self.lock:
            state = self.sources[source]
            first_candidate = state.latest_chunk is None
            session_changed = (
                state.latest_chunk is not None
                and chunk.session_id != state.latest_chunk.session_id
            )
            if not session_changed and chunk.sequence <= state.latest_sequence:
                return
            if session_changed:
                state.latest_action = None
                state.latest_sequence = -1
            if len(chunk.actions) != 1:
                self._reject_source_chunk_locked(
                    source,
                    f"{source} chunks must contain exactly one action",
                    now,
                    blocking=True,
                )
                return
            if (
                source == "teleop"
                and state.latest_chunk is not None
                and not session_changed
            ):
                previous = state.latest_chunk.actions[0]
                current = chunk.actions[0]
                for side in ("left", "right"):
                    old = previous.command(side)
                    new = current.command(side)
                    translation_jump = float(
                        np.linalg.norm(new.position - old.position)
                    )
                    rotation_jump = quaternion_angle(
                        new.quaternion, old.quaternion
                    )
                    if (
                        translation_jump > self.args.max_source_jump
                        or rotation_jump > self.args.max_source_rotation_jump
                    ):
                        reason = (
                            f"teleop {side} tracking jump: "
                            f"{translation_jump:.3f} m, {rotation_jump:.3f} rad"
                        )
                        state.invalid_reason = reason
                        if self.selected_mode == source:
                            self._pause_locked(reason, now)
                        return

            state.latest_chunk = chunk
            state.latest_sequence = chunk.sequence
            state.chunk_received_at = now
            state.latest_created_at_ns = int(chunk.created_at_ns)
            state.invalid_reason = ""
            state.accepted_chunks += 1
            state.last_rejection_reason = ""
            if source == "teleop":
                # VR packets are state samples, not a trajectory. Queueing
                # every sample makes a faster headset stream build seconds of
                # stale controller motion behind the 60 Hz API loop. Keep
                # only the newest complete action so the robot follows the
                # current hand pose with bounded latency.
                dropped = int(state.latest_action is not None)
                state.latest_action = chunk.actions[0]
                state.superseded_actions += dropped
            else:
                # Policy is a single-action target stream. It is forwarded from
                # this callback rather than consumed by the 60 Hz teleop loop.
                state.latest_action = None
            if source == "policy" and first_candidate:
                LOGGER.info(
                    "[POLICY] 收到动作：session=%s sequence=%d actions=%d",
                    chunk.session_id[:12],
                    chunk.sequence,
                    len(chunk.actions),
                )
            if session_changed and self.selected_mode == source:
                self._pause_locked(f"{source} source session restarted", now)
                return
            if (
                source == "policy"
                and self.selected_mode == source
                and self.phase == "active"
            ):
                valid, reason = self._runtime_ready(source, now)
                if not valid:
                    self._pause_locked(reason, now)
                    return
                if not self._fsm_transition_reached(FSM_READY):
                    self._pause_locked("robot FSM is no longer READY", now)
                    return
                self._publish_policy_action_locked(chunk.actions[0])

    def _on_mode_request(self, msg) -> None:
        requested = msg.data.strip().lower()
        now = time.monotonic()
        with self.lock:
            if requested not in topics.MODES:
                LOGGER.warning("Ignoring unsupported HIL mode request: %r", msg.data)
                return
            if requested == "pause":
                LOGGER.info("[切换] 收到 PAUSE 请求：正在锁停机器人")
                self._pause_locked("pause requested", now)
                return

            LOGGER.info("[切换请求] 收到 %s 请求，正在检查接管条件", requested.upper())
            valid, reason = self._base_ready(now)
            if not valid:
                self._pause_locked(f"rejected {requested}: {reason}", now)
                return

            if requested == "policy":
                self.policy_request_received_at = now
                self._reset_policy_source_locked()

            self.pending_mode = requested
            self.selected_mode = "pause"
            self.phase = "waiting_for_pause"
            self.phase_started_at = now
            self.last_reason = f"switching to {requested}"
            self._publish_enable_locked(0.0)
            if requested == "policy":
                LOGGER.info(
                    "[切换] 收到 POLICY 请求：已清空旧 policy 动作，正在锁停机器人"
                )
            else:
                LOGGER.info("[切换] 收到 TELEOP 请求：正在锁停并准备遥操作锚定")

    def _reset_policy_source_locked(self) -> None:
        state = self.sources["policy"]
        state.latest_chunk = None
        state.chunk_received_at = 0.0
        state.latest_created_at_ns = 0
        state.latest_sequence = -1
        state.latest_action = None
        state.invalid_reason = ""
        state.last_rejection_reason = ""

    def _mode_configuration_valid(self, now: float) -> tuple[bool, str]:
        if self.args.skip_mode_check:
            return True, "mode check disabled"
        if self.current_mode.age(now) > self.args.mode_timeout:
            return False, "/api/current_mode is missing or stale"
        mode = self.current_mode.value
        checks = (
            (mode.get("meta_mode") == 1, "robot is not in API mode"),
            (mode.get("left_arm_control_mode") == 0, "left arm is not in endpoint mode"),
            (mode.get("right_arm_control_mode") == 0, "right arm is not in endpoint mode"),
            (bool(mode.get("enable_left_arm")), "left arm is disabled"),
            (bool(mode.get("enable_right_arm")), "right arm is disabled"),
        )
        for passed, reason in checks:
            if not passed:
                return False, reason
        return True, "ok"

    def _feedback_ready(self, now: float) -> tuple[bool, str]:
        for side in ("left", "right"):
            if self.robot_poses[side].age(now) > self.args.feedback_timeout:
                return False, f"{side} current_ee_pose is missing or stale"
        if self.fsm_state.age(now) > self.args.feedback_timeout:
            return False, "/fsm_state is missing or stale"
        if self.fsm_state.value == FSM_ERROR:
            return False, "robot FSM is ERROR"
        return True, "ok"

    def _source_ready(self, source: str, now: float) -> tuple[bool, str]:
        state = self.sources[source]
        if state.invalid_reason:
            return False, state.invalid_reason
        if state.latest_chunk is None or state.chunk_received_at <= 0.0:
            return False, f"{source} action chunk is missing"
        source_timeout = (
            self.args.policy_source_timeout
            if source == "policy"
            else self.args.teleop_source_timeout
        )
        if now - state.chunk_received_at > source_timeout:
            return False, f"{source} action chunk is stale"
        if source == "teleop" and not self.args.skip_teleop_status_check:
            status = self.teleop_server_status
            if status.age(now) > self.args.teleop_status_timeout:
                return False, "teleop server status is missing or stale"
            if not status.value.get("pose_transport_connected"):
                return False, "VR pose stream is disconnected"
            if not status.value.get("tracking_valid"):
                return False, "VR controller tracking is invalid"
            pose_age = status.value.get("pose_age_s")
            if pose_age is None or pose_age > self.args.teleop_source_timeout:
                return False, "VR pose stream is stale"
        return True, "ok"

    def _base_ready(self, now: float) -> tuple[bool, str]:
        for check in (
            self._mode_configuration_valid(now),
            self._api_ownership_valid(now),
            self._feedback_ready(now),
        ):
            if not check[0]:
                return check
        return True, "ok"

    def _runtime_ready(self, source: str, now: float) -> tuple[bool, str]:
        valid, reason = self._base_ready(now)
        if not valid:
            return valid, reason
        return self._source_ready(source, now)

    def _policy_handover_ready(self) -> tuple[bool, str]:
        state = self.sources["policy"]
        if state.latest_chunk is None:
            return False, "policy action chunk is missing"
        if state.chunk_received_at <= self.policy_request_received_at:
            return False, "policy chunk predates the handover request"
        return True, "fresh policy action received after handover request"

    def _activation_ready(self, source: str, now: float) -> tuple[bool, str]:
        valid, reason = self._runtime_ready(source, now)
        if not valid:
            return valid, reason
        if source == "policy":
            return self._policy_handover_ready()
        return True, "ok"

    def _api_ownership_valid(self, now: float) -> tuple[bool, str]:
        if self.args.skip_ownership_check:
            return True, "ownership check disabled"
        if self.api_ownership.age(now) > self.args.ownership_timeout:
            return False, "API publisher ownership check is stale"
        return self.api_ownership.value

    def _ownership_tick(self) -> None:
        own_name = self.node.get_name()
        own_namespace = self.node.get_namespace()
        command_topics = (
            topics.API_LEFT_POSE,
            topics.API_RIGHT_POSE,
            topics.API_LEFT_GRIPPER,
            topics.API_RIGHT_GRIPPER,
            topics.API_FSM_ENABLE,
        )
        conflicts = []
        for topic in command_topics:
            publishers = self.node.get_publishers_info_by_topic(topic)
            own_publishers = [
                info
                for info in publishers
                if info.node_name == own_name and info.node_namespace == own_namespace
            ]
            if len(publishers) != 1 or len(own_publishers) != 1:
                names = sorted(
                    f"{info.node_namespace.rstrip('/')}/{info.node_name}"
                    for info in publishers
                )
                conflicts.append(f"{topic} publishers={names}")
        result = (
            (False, "foreign or duplicate API publishers: " + "; ".join(conflicts))
            if conflicts
            else (True, "ok")
        )
        now = time.monotonic()
        with self.lock:
            self.api_ownership = TimedValue(result, now)
            if not result[0] and self.phase != "paused":
                self._pause_locked(result[1], now)

    def _pause_locked(self, reason: str, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        changed = self.phase != "paused" or self.last_reason != reason
        self.phase = "paused"
        self.selected_mode = "pause"
        self.pending_mode = None
        self.phase_started_at = now
        self.last_reason = reason
        self.teleop_raw_anchors.clear()
        self.teleop_robot_anchors.clear()
        for state in self.sources.values():
            state.latest_action = None
        self._publish_enable_locked(0.0)
        if changed:
            LOGGER.warning("[安全暂停] 机器人已锁停：%s", reason)

    def _publish_enable_locked(self, value: float) -> None:
        if value == 0.0:
            self._last_disable_at = time.monotonic()
        if self.args.dry_run:
            return
        self.enable_publisher.publish(self.Float32(data=float(value)))

    def _fsm_transition_reached(self, expected_state: int) -> bool:
        """Dry-run exercises routing without waiting for commands it never sends."""
        return self.args.dry_run or self.fsm_state.value == expected_state

    def _begin_source_locked(self, source: str, now: float) -> None:
        valid, reason = self._activation_ready(source, now)
        if not valid:
            self._pause_locked(f"cannot activate {source}: {reason}", now)
            return

        for side in ("left", "right"):
            robot = self.robot_poses[side].value
            self.last_outputs[side] = PoseArray(
                position=robot.position.copy(), quaternion=robot.quaternion.copy()
            )
        if source == "teleop":
            self._reanchor_teleop_locked()

        # A source may have been producing while paused. Teleop starts from its
        # newest sample; policy is forwarded only after FSM READY.
        state = self.sources[source]
        state.latest_action = (
            state.latest_chunk.actions[0] if source == "teleop" else None
        )

        self.selected_mode = source
        self.pending_mode = None
        self.phase = "starting"
        self.phase_started_at = now
        self.last_reason = f"{source} selected; waiting for FSM READY"
        self._publish_outputs_locked(now)
        self._publish_enable_locked(1.0)
        if source == "policy":
            LOGGER.info("[切换] policy 候选通过：%s", reason)
            LOGGER.info("[切换] 正在启用 POLICY，等待机器人 FSM READY")
        else:
            LOGGER.info("[切换] 遥操作已从当前末端位姿锚定，等待机器人 FSM READY")

    def _reanchor_teleop_locked(self) -> None:
        first_action = self.sources["teleop"].latest_chunk.actions[0]
        for side in ("left", "right"):
            raw = first_action.command(side)
            robot = self.robot_poses[side].value
            self.teleop_raw_anchors[side] = PoseArray(
                raw.position.copy(), raw.quaternion.copy()
            )
            self.teleop_robot_anchors[side] = PoseArray(
                robot.position.copy(), robot.quaternion.copy()
            )
            self.last_outputs[side] = PoseArray(
                robot.position.copy(), robot.quaternion.copy()
            )

    def _teleop_candidate_pose_locked(self, side: str, action: ActionFrame) -> PoseArray:
        raw = action.command(side)
        raw_anchor = self.teleop_raw_anchors[side]
        robot_anchor = self.teleop_robot_anchors[side]
        position, quaternion = anchored_pose(
            raw.position,
            raw.quaternion,
            raw_anchor.position,
            raw_anchor.quaternion,
            robot_anchor.position,
            robot_anchor.quaternion,
            position_scale=self.args.teleop_position_scale,
        )
        displacement = position - robot_anchor.position
        distance = float(np.linalg.norm(displacement))
        if distance > self.args.max_teleop_displacement:
            position = robot_anchor.position + displacement * (
                self.args.max_teleop_displacement / distance
            )
        position = clamp_arm_reach(position, self.args.max_arm_reach)
        return PoseArray(position, quaternion)

    def _publish_policy_action_locked(self, action: ActionFrame) -> None:
        """Forward one active-policy action without motion shaping."""
        for side in ("left", "right"):
            command = action.command(side)
            output = PoseArray(command.position.copy(), command.quaternion.copy())
            self.last_outputs[side] = output
            self.last_grippers[side] = command.gripper
            if not self.args.dry_run:
                self.api_pose_publishers[side].publish(self._array_to_pose(output))
                self.api_gripper_publishers[side].publish(
                    self.Float32(data=command.gripper)
                )

    def _publish_policy_handover_locked(self, action: ActionFrame) -> bool:
        """Move toward the latest policy target at bounded speed during handback."""

        dt = 1.0 / self.args.control_hz
        command_reached = True
        for side in ("left", "right"):
            command = action.command(side)
            previous = self.last_outputs[side]
            if previous is None:
                previous = self.robot_poses[side].value
            position, quaternion = limit_pose_step(
                previous.position,
                previous.quaternion,
                command.position,
                command.quaternion,
                self.args.max_translation_speed * dt,
                self.args.max_rotation_speed * dt,
            )
            output = PoseArray(position, quaternion)
            self.last_outputs[side] = output
            command_reached = command_reached and (
                float(np.linalg.norm(output.position - command.position)) <= 1e-9
                and quaternion_angle(output.quaternion, command.quaternion) <= 1e-9
            )

            desired_gripper = command.gripper
            rate = (
                self.args.max_gripper_release_rate
                if desired_gripper < self.last_grippers[side]
                else self.args.max_gripper_rate
            )
            max_gripper_step = rate * dt
            gripper_delta = float(
                np.clip(
                    desired_gripper - self.last_grippers[side],
                    -max_gripper_step,
                    max_gripper_step,
                )
            )
            self.last_grippers[side] = float(
                np.clip(self.last_grippers[side] + gripper_delta, 0.0, 1.0)
            )
            command_reached = command_reached and (
                abs(self.last_grippers[side] - desired_gripper) <= 1e-9
            )

            if not self.args.dry_run:
                self.api_pose_publishers[side].publish(self._array_to_pose(output))
                self.api_gripper_publishers[side].publish(
                    self.Float32(data=self.last_grippers[side])
                )

        feedback_poses = {
            side: (
                self.last_outputs[side]
                if self.args.dry_run
                else self.robot_poses[side].value
            )
            for side in ("left", "right")
        }
        feedback_reached, reason = policy_handover_target_reached(
            action,
            feedback_poses,
            position_limit=self.args.policy_handover_position_limit,
            rotation_limit=self.args.policy_handover_rotation_limit,
        )
        self.last_reason = f"policy handover tracking: {reason}"
        return command_reached and feedback_reached

    def _publish_outputs_locked(self, now: float) -> None:
        source = self.selected_mode
        if source not in ("policy", "teleop"):
            return
        ready, reason = self._source_ready(source, now)
        if not ready:
            self._pause_locked(reason, now)
            return

        if source == "policy":
            # Active policy output is event-driven in _on_source_chunk. During
            # handover, the 60 Hz loop follows the newest target with bounded
            # steps until the measured endpoints have caught up.
            if self.phase == "active":
                return
            if self.phase == "policy_handover":
                action = self.sources["policy"].latest_chunk.actions[0]
                if self._publish_policy_handover_locked(action):
                    self.phase = "active"
                    self.phase_started_at = now
                    self.last_reason = "policy active"
                    LOGGER.info(
                        "[切换] POLICY 平滑接管完成，恢复按 policy 原频率直通"
                    )
                return

        dt = 1.0 / self.args.control_hz
        state = self.sources[source]
        action = state.latest_action if self.phase == "active" else None
        for side in ("left", "right"):
            previous = self.last_outputs[side]
            if previous is None:
                previous = self.robot_poses[side].value
            if self.phase == "starting" or action is None:
                position = previous.position.copy()
                quaternion = previous.quaternion.copy()
            else:
                candidate = self._teleop_candidate_pose_locked(side, action)
                position, quaternion = limit_pose_step(
                    previous.position,
                    previous.quaternion,
                    candidate.position,
                    candidate.quaternion,
                    self.args.max_translation_speed * dt,
                    self.args.max_rotation_speed * dt,
                )
            output = PoseArray(position, quaternion)
            self.last_outputs[side] = output

            desired_gripper = (
                self.last_grippers[side]
                if self.phase == "starting" or action is None
                else action.command(side).gripper
            )
            rate = (
                self.args.max_gripper_release_rate
                if desired_gripper < self.last_grippers[side]
                else self.args.max_gripper_rate
            )
            max_gripper_step = rate * dt
            gripper_delta = float(
                np.clip(
                    desired_gripper - self.last_grippers[side],
                    -max_gripper_step,
                    max_gripper_step,
                )
            )
            self.last_grippers[side] = float(
                np.clip(self.last_grippers[side] + gripper_delta, 0.0, 1.0)
            )

            if not self.args.dry_run:
                self.api_pose_publishers[side].publish(self._array_to_pose(output))
                self.api_gripper_publishers[side].publish(
                    self.Float32(data=self.last_grippers[side])
                )

    def _control_tick(self) -> None:
        now = time.monotonic()
        with self.lock:
            if self.phase == "paused":
                return
            valid, reason = self._mode_configuration_valid(now)
            if not valid:
                self._pause_locked(reason, now)
                return
            valid, reason = self._api_ownership_valid(now)
            if not valid:
                self._pause_locked(reason, now)
                return
            valid, reason = self._feedback_ready(now)
            if not valid:
                self._pause_locked(reason, now)
                return

            if self.phase == "waiting_for_pause":
                if now - self._last_disable_at >= 0.1:
                    self._publish_enable_locked(0.0)
                if self._fsm_transition_reached(FSM_PAUSE):
                    self.phase = "waiting_for_source"
                    self.phase_started_at = now
                    self.last_reason = (
                        f"waiting for fresh {self.pending_mode} action chunk"
                    )
                    LOGGER.info(
                        "[切换] 机器人已进入 PAUSE：等待 %s 新动作",
                        self.pending_mode.upper(),
                    )
                elif now - self.phase_started_at > self.args.pause_timeout:
                    self._pause_locked("timed out waiting for FSM PAUSE", now)
                return

            if self.phase == "waiting_for_source":
                valid, reason = self._activation_ready(self.pending_mode, now)
                if valid:
                    self._begin_source_locked(self.pending_mode, now)
                elif now - self.phase_started_at > self.args.source_wait_timeout:
                    self._pause_locked(
                        f"timed out waiting for {self.pending_mode}: {reason}", now
                    )
                else:
                    self.last_reason = f"waiting for {self.pending_mode}: {reason}"
                return

            self._publish_outputs_locked(now)
            if self.phase == "starting":
                if self._fsm_transition_reached(FSM_READY):
                    if self.selected_mode == "teleop":
                        self._reanchor_teleop_locked()
                    elif self.selected_mode == "policy":
                        valid, reason = self._policy_handover_ready()
                        if not valid:
                            self._pause_locked(
                                f"policy handover rejected: {reason}", now
                            )
                            return
                    state = self.sources[self.selected_mode]
                    state.latest_action = (
                        state.latest_chunk.actions[0]
                        if self.selected_mode == "teleop"
                        else None
                    )
                    self.phase_started_at = now
                    if self.selected_mode == "policy":
                        self.phase = "policy_handover"
                        _, target_reason = policy_handover_target_reached(
                            state.latest_chunk.actions[0],
                            {
                                side: self.robot_poses[side].value
                                for side in ("left", "right")
                            },
                            position_limit=self.args.policy_handover_position_limit,
                            rotation_limit=self.args.policy_handover_rotation_limit,
                        )
                        self.last_reason = f"policy handover tracking: {target_reason}"
                        LOGGER.info(
                            "[切换] POLICY FSM READY：开始以 %.3f m/s、%.3f rad/s "
                            "平滑追踪最新目标（%s）",
                            self.args.max_translation_speed,
                            self.args.max_rotation_speed,
                            target_reason,
                        )
                    else:
                        self.phase = "active"
                        self.last_reason = "teleop active"
                        LOGGER.info("[切换] TELEOP 已接管，机器人 FSM READY")
                elif now - self.phase_started_at > self.args.ready_timeout:
                    self._pause_locked("timed out waiting for FSM READY", now)
            elif (
                self.phase in ("active", "policy_handover")
                and not self.args.dry_run
                and self.fsm_state.value == FSM_PAUSE
            ):
                self._pause_locked("robot FSM unexpectedly entered PAUSE", now)

    def _heartbeat_tick(self) -> None:
        now = time.monotonic()
        with self.lock:
            if self.phase in (
                "starting",
                "policy_handover",
                "active",
            ) and self.selected_mode in ("policy", "teleop"):
                valid, reason = self._runtime_ready(self.selected_mode, now)
                if valid:
                    self._publish_enable_locked(1.0)
                else:
                    self._pause_locked(reason, now)
            else:
                self._publish_enable_locked(0.0)

    def _status_tick(self) -> None:
        now = time.monotonic()
        with self.lock:
            status = {
                "phase": self.phase,
                "selected_mode": self.selected_mode,
                "pending_mode": self.pending_mode,
                "reason": self.last_reason,
                "fsm_state": self.fsm_state.value,
                "fsm_age_s": round(self.fsm_state.age(now), 3),
                "mode_config_valid": self._mode_configuration_valid(now)[0],
                "api_ownership_valid": self._api_ownership_valid(now)[0],
                "robot_pose_age_s": {
                    side: round(self.robot_poses[side].age(now), 3) for side in ("left", "right")
                },
                "source_age_s": {
                    source: (
                        None
                        if self.sources[source].chunk_received_at <= 0.0
                        else round(now - self.sources[source].chunk_received_at, 3)
                    )
                    for source in ("policy", "teleop")
                },
                "source_action_age_s": {
                    source: (
                        None
                        if self.sources[source].latest_created_at_ns <= 0
                        else round(
                            max(
                                0.0,
                                (time.time_ns() - self.sources[source].latest_created_at_ns)
                                / 1e9,
                            ),
                            3,
                        )
                    )
                    for source in ("policy", "teleop")
                },
                "source_sequence": {
                    source: self.sources[source].latest_sequence
                    for source in ("policy", "teleop")
                },
                "source_session": {
                    source: (
                        ""
                        if self.sources[source].latest_chunk is None
                        else self.sources[source].latest_chunk.session_id[:12]
                    )
                    for source in ("policy", "teleop")
                },
                "source_queue_depth": {
                    source: int(self.sources[source].latest_action is not None)
                    for source in ("policy", "teleop")
                },
                "source_superseded_actions": {
                    source: self.sources[source].superseded_actions
                    for source in ("policy", "teleop")
                },
                "source_accepted_chunks": {
                    source: self.sources[source].accepted_chunks
                    for source in ("policy", "teleop")
                },
                "source_rejected_chunks": {
                    source: self.sources[source].rejected_chunks
                    for source in ("policy", "teleop")
                },
                "source_last_rejection": {
                    source: self.sources[source].last_rejection_reason
                    for source in ("policy", "teleop")
                },
                "teleop_status_age_s": round(self.teleop_server_status.age(now), 3),
                "dry_run": self.args.dry_run,
            }
            encoded = json.dumps(status, sort_keys=True, allow_nan=True)
            self.status_publisher.publish(self.String(data=encoded))

    def close(self) -> None:
        if self.rclpy.ok():
            try:
                with self.lock:
                    self._pause_locked("supervisor shutdown")
                    if not self.args.dry_run:
                        for _ in range(3):
                            self.enable_publisher.publish(self.Float32(data=0.0))
            except Exception as exc:
                # The ROS context can become invalid between ok() and publish
                # when DDS or another process initiates shutdown concurrently.
                LOGGER.warning("Supervisor shutdown publish skipped: %s", exc)
        try:
            self.node.destroy_node()
        except Exception as exc:
            LOGGER.warning("Supervisor node cleanup skipped: %s", exc)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-hz", type=float, default=60.0)
    parser.add_argument("--heartbeat-hz", type=float, default=20.0)
    parser.add_argument("--policy-source-timeout", type=float, default=3.0)
    parser.add_argument("--teleop-source-timeout", type=float, default=0.50)
    parser.add_argument("--max-chunk-actions", type=int, default=240)
    parser.add_argument("--feedback-timeout", type=float, default=0.5)
    parser.add_argument("--mode-timeout", type=float, default=3.0)
    parser.add_argument("--ownership-timeout", type=float, default=2.0)
    parser.add_argument("--teleop-status-timeout", type=float, default=1.0)
    parser.add_argument("--pause-timeout", type=float, default=2.0)
    parser.add_argument("--source-wait-timeout", type=float, default=10.0)
    parser.add_argument("--ready-timeout", type=float, default=5.0)
    parser.add_argument("--max-arm-reach", type=float, default=0.55)
    parser.add_argument("--max-translation-speed", type=float, default=0.25)
    parser.add_argument("--max-rotation-speed", type=float, default=1.2)
    parser.add_argument("--max-gripper-rate", type=float, default=2.0)
    parser.add_argument("--max-gripper-release-rate", type=float, default=10.0)
    parser.add_argument("--max-source-jump", type=float, default=0.15)
    parser.add_argument("--max-source-rotation-jump", type=float, default=1.0)
    parser.add_argument(
        "--policy-handover-position-limit", type=float, default=0.03
    )
    parser.add_argument(
        "--policy-handover-rotation-limit", type=float, default=0.262
    )
    parser.add_argument("--max-teleop-displacement", type=float, default=0.40)
    parser.add_argument("--teleop-position-scale", type=float, default=1.0)
    parser.add_argument(
        "--skip-mode-check",
        action="store_true",
        help="Lab diagnostics only: do not require /api/current_mode confirmation.",
    )
    parser.add_argument(
        "--skip-ownership-check",
        action="store_true",
        help="Lab diagnostics only: do not reject other /api command publishers.",
    )
    parser.add_argument(
        "--skip-teleop-status-check",
        action="store_true",
        help="Lab diagnostics only: do not require second-S100 pose status.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all subscriptions/state transitions without publishing /api commands.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.max_chunk_actions < 1 or args.max_chunk_actions > 1000:
        parser.error("--max-chunk-actions must be in [1, 1000]")
    for name in (
        "control_hz",
        "heartbeat_hz",
        "policy_source_timeout",
        "teleop_source_timeout",
        "feedback_timeout",
        "mode_timeout",
        "ownership_timeout",
        "teleop_status_timeout",
        "pause_timeout",
        "source_wait_timeout",
        "ready_timeout",
        "max_arm_reach",
        "max_translation_speed",
        "max_rotation_speed",
        "max_gripper_rate",
        "max_gripper_release_rate",
        "max_source_jump",
        "max_source_rotation_jump",
        "policy_handover_position_limit",
        "policy_handover_rotation_limit",
        "max_teleop_displacement",
        "teleop_position_scale",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    import rclpy

    rclpy.init()
    supervisor = HILSupervisor(args)
    try:
        rclpy.spin(supervisor.node)
    except KeyboardInterrupt:
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        supervisor.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
