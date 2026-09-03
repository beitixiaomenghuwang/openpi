#!/usr/bin/env python3
"""ROS2/RTP interface for bimanual TeleAvatar V2 end-effector control."""

from __future__ import annotations

import pathlib
import sys
from threading import Lock
import time

from geometry_msgs.msg import Pose
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.teleavatar_v2.rtp_video_interface import RTPH265VideoInterface  # noqa: E402

_EE_STATE_OFFSET = {"left": 48, "right": 55}
_GRIPPER_STATE_INDEX = {"left": 7, "right": 15}
_POLICY_TO_RTP_VIEW = {
    "head_camera": "head_left_eye",
    "left_color": "left_wrist_right_eye",
    "right_color": "right_wrist_left_eye",
}


def _pose_to_array(pose: Pose) -> np.ndarray:
    return np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float32,
    )


def _quaternion_angle(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest angular distance between two xyzw quaternions."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first.shape != (4,) or second.shape != (4,) or first_norm < 1e-7 or second_norm < 1e-7:
        raise ValueError("Quaternion comparison requires two non-zero 4D quaternions")
    dot = float(np.dot(first / first_norm, second / second_norm))
    return float(2.0 * np.arccos(np.clip(abs(dot), 0.0, 1.0)))


def _quaternion_nlerp(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate two xyzw quaternions on the shortest path."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if float(np.dot(first, second)) < 0.0:
        second = -second
    value = (1.0 - alpha) * first + alpha * second
    norm = float(np.linalg.norm(value))
    if norm < 1e-7:
        raise ValueError("Quaternion interpolation produced a degenerate value")
    return value / norm


def _rot6d_to_quaternion(rot6d: np.ndarray) -> np.ndarray:
    """Convert row-wise rotation-6D to a normalized ROS xyzw quaternion."""
    value = np.asarray(rot6d, dtype=np.float64)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Expected finite rotation-6D shape (6,), got {value.shape}")

    row_1 = value[:3]
    row_2 = value[3:]
    norm_1 = float(np.linalg.norm(row_1))
    if norm_1 < 1e-7:
        raise ValueError("Rotation-6D first row is degenerate")
    row_1 /= norm_1
    row_2 -= np.dot(row_1, row_2) * row_1
    norm_2 = float(np.linalg.norm(row_2))
    if norm_2 < 1e-7:
        raise ValueError("Rotation-6D second row is degenerate")
    row_2 /= norm_2
    matrix = np.stack((row_1, row_2, np.cross(row_1, row_2)), axis=0)
    return _matrix_to_quaternion(matrix)


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an xyzw quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (m[2, 1] - m[1, 2]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
                (m[1, 0] - m[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = np.diag(m)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0))
            quaternion = np.array(
                [0.25 * scale, (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale, (m[2, 1] - m[1, 2]) / scale]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0))
            quaternion = np.array(
                [(m[0, 1] + m[1, 0]) / scale, 0.25 * scale, (m[1, 2] + m[2, 1]) / scale, (m[0, 2] - m[2, 0]) / scale]
            )
        else:
            scale = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0))
            quaternion = np.array(
                [(m[0, 2] + m[2, 0]) / scale, (m[1, 2] + m[2, 1]) / scale, 0.25 * scale, (m[1, 0] - m[0, 1]) / scale]
            )

    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-7:
        raise ValueError("Rotation matrix produced a degenerate quaternion")
    return (quaternion / norm).astype(np.float32)


class TeleavatarV2EEInterface(Node):
    """Collect bimanual EE observations and publish platform API commands.

    The robot does not expose gripper position feedback in the API used here.
    The raw-state gripper entry therefore tracks the last command sent by this
    process and starts from the two configured initial trigger values. They
    must match the physical gripper states when deployment starts.
    """

    def __init__(
        self,
        *,
        rtp_port: int = 8890,
        rtp_payload: int = 96,
        rtp_decoder: str = "nvh265dec max-display-delay=0",
        sensor_timeout: float = 1.0,
        control_frequency: float = 45.0,
        interp_frequency: float = 200.0,
        interpolate: bool = True,
        initial_left_gripper_trigger: float = 0.0,
        initial_right_gripper_trigger: float = 0.0,
        node_name: str = "teleavatar_v2_ee_openpi_interface",
    ) -> None:
        super().__init__(node_name)
        if sensor_timeout <= 0.0:
            raise ValueError("sensor_timeout must be positive")
        if control_frequency <= 0.0:
            raise ValueError("control_frequency must be positive")
        if interp_frequency <= 0.0:
            raise ValueError("interp_frequency must be positive")
        self.sensor_timeout = float(sensor_timeout)
        self._lock = Lock()
        self._latest_poses: dict[str, Pose] = {}
        self._pose_timestamps: dict[str, float] = {}
        self._last_gripper_triggers = {
            "left": float(np.clip(initial_left_gripper_trigger, 0.0, 1.0)),
            "right": float(np.clip(initial_right_gripper_trigger, 0.0, 1.0)),
        }
        self._cmd_lock = Lock()
        self._publish_lock = Lock()
        self._output_enabled = False
        self._interpolate = bool(interpolate)
        self._ctrl_period = 1.0 / float(control_frequency)
        self._interp_period = 1.0 / float(interp_frequency)
        self._ramp_from: np.ndarray | None = None
        self._ramp_to: np.ndarray | None = None
        self._ramp_t0: float | None = None
        self._last_cmd_action: np.ndarray | None = None
        self._have_target = False
        self._target_generation = 0
        self._enable_counter = 0

        self._video = RTPH265VideoInterface(
            port=rtp_port,
            payload=rtp_payload,
            decoder=rtp_decoder,
        )
        self._video.start()

        self._pose_topics = {arm: f"/{arm}_arm/current_ee_pose" for arm in ("left", "right")}
        self._pose_publishers = {
            arm: self.create_publisher(Pose, f"/api/{arm}_arm/target_pose", 10) for arm in ("left", "right")
        }
        self._gripper_publishers = {
            arm: self.create_publisher(Float32, f"/api/{arm}_gripper/cmd", 10) for arm in ("left", "right")
        }
        self._enable_publisher = self.create_publisher(Float32, "/api/fsm/enable", 10)
        for arm in ("left", "right"):
            self.create_subscription(
                Pose,
                self._pose_topics[arm],
                lambda message, side=arm: self._on_pose(message, side),
                10,
            )
        self._interp_timer = self.create_timer(self._interp_period, self._interp_publish) if self._interpolate else None

        self.get_logger().info(
            f"Bimanual EE interface ready: poses={list(self._pose_topics.values())}, RTP=udp/{rtp_port}, "
            f"initial gripper triggers={self._last_gripper_triggers}"
        )
        if self._interpolate:
            self.get_logger().info(
                f"Pose interpolation ON: {control_frequency:.1f} Hz targets -> {interp_frequency:.1f} Hz publish "
                f"(ramp {self._ctrl_period * 1000.0:.1f} ms, FSM heartbeat about {interp_frequency / 4.0:.1f} Hz)"
            )
        else:
            self.get_logger().info(
                f"Pose interpolation OFF: publishing targets directly at about {control_frequency:.1f} Hz "
                "(FSM heartbeat every 4 commands)"
            )

    def _on_pose(self, message: Pose, arm: str) -> None:
        pose = _pose_to_array(message)
        quaternion_norm = float(np.linalg.norm(pose[3:7]))
        if not np.all(np.isfinite(pose)) or quaternion_norm < 1e-6:
            self.get_logger().error(f"Rejected invalid {arm} current end-effector pose", throttle_duration_sec=1.0)
            return
        with self._lock:
            self._latest_poses[arm] = message
            self._pose_timestamps[arm] = time.time()

    def sensor_errors(self) -> list[str]:
        """Return missing/stale input descriptions without copying video frames."""
        now = time.time()
        errors: list[str] = []
        if self._video.stream_ended():
            errors.append("RTP pipeline stopped")

        timestamps = self._video.get_image_timestamps()
        for view in _POLICY_TO_RTP_VIEW.values():
            timestamp = timestamps.get(view)
            if timestamp is None:
                errors.append(f"video:{view} missing")
            elif now - timestamp > self.sensor_timeout:
                errors.append(f"video:{view} stale ({now - timestamp:.2f}s)")

        with self._lock:
            pose_timestamps = dict(self._pose_timestamps)
        for arm, topic in self._pose_topics.items():
            pose_timestamp = pose_timestamps.get(arm)
            if pose_timestamp is None:
                errors.append(f"pose:{topic} missing")
            elif now - pose_timestamp > self.sensor_timeout:
                errors.append(f"pose:{topic} stale ({now - pose_timestamp:.2f}s)")
        return errors

    def ee_quaternion_target_errors(
        self,
        action: np.ndarray,
        *,
        max_position_error: float = 0.20,
        max_orientation_error: float = 0.80,
    ) -> list[str]:
        """Return safety violations between a 16D target and measured EE poses.

        Each arm is ``xyz + quaternion xyzw + gripper trigger``. This check
        deliberately compares against the latest measured current pose, so an
        unexpected model jump is rejected before the target is published. A
        threshold of zero disables that component.
        """
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (16,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Expected a finite 16D quaternion action, got shape {value.shape}")
        if max_position_error < 0.0 or max_orientation_error < 0.0:
            raise ValueError("EE safety thresholds cannot be negative")
        if max_position_error == 0.0 and max_orientation_error == 0.0:
            return []

        with self._lock:
            current_poses = dict(self._latest_poses)

        errors: list[str] = []
        for arm, start in (("left", 0), ("right", 8)):
            current_pose = current_poses.get(arm)
            if current_pose is None:
                errors.append(f"{arm} current EE pose is unavailable")
                continue
            current = _pose_to_array(current_pose)
            target_position = value[start : start + 3]
            target_quaternion = value[start + 3 : start + 7]
            position_error = float(np.linalg.norm(target_position - current[:3]))
            orientation_error = _quaternion_angle(current[3:7], target_quaternion)
            violations = []
            if max_position_error > 0.0 and position_error > max_position_error:
                violations.append(f"position {position_error:.3f}m > {max_position_error:.3f}m")
            if max_orientation_error > 0.0 and orientation_error > max_orientation_error:
                violations.append(f"orientation {orientation_error:.3f}rad > {max_orientation_error:.3f}rad")
            if violations:
                errors.append(f"{arm} target/current: " + ", ".join(violations))
        return errors

    def wait_for_initial_data(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        next_log = 0.0
        while time.monotonic() < deadline:
            errors = self.sensor_errors()
            if not errors:
                self.get_logger().info("RTP views and current end-effector pose received")
                return True
            if time.monotonic() >= next_log:
                self.get_logger().info("Waiting for inputs: " + "; ".join(errors))
                next_log = time.monotonic() + 2.0
            time.sleep(0.05)
        self.get_logger().error("Timed out waiting for inputs: " + "; ".join(self.sensor_errors()))
        return False

    def get_policy_observation(self, prompt: str) -> dict:
        errors = self.sensor_errors()
        if errors:
            raise RuntimeError("Observation unavailable: " + "; ".join(errors))

        policy_views = tuple(_POLICY_TO_RTP_VIEW.values())
        images, _timestamps = self._video.get_latest_images_with_timestamps(policy_views)
        with self._lock:
            poses = dict(self._latest_poses)
            gripper_triggers = dict(self._last_gripper_triggers)

        # Raw converter-compatible prefix: positions/velocities/efforts (48)
        # followed by left/right EE pose7. The EE transform consumes both
        # poses and gripper positions; unused joint fields stay 0.
        state = np.zeros(62, dtype=np.float32)
        for arm in ("left", "right"):
            state[_EE_STATE_OFFSET[arm] : _EE_STATE_OFFSET[arm] + 7] = _pose_to_array(poses[arm])
            # Raw gripper position is openness (1=open), while API/model
            # trigger is the inverse (0=open).
            state[_GRIPPER_STATE_INDEX[arm]] = 1.0 - gripper_triggers[arm]

        return {
            "observation/state": state,
            "observation/images/left_color": images[_POLICY_TO_RTP_VIEW["left_color"]],
            "observation/images/right_color": images[_POLICY_TO_RTP_VIEW["right_color"]],
            "observation/images/head_camera": images[_POLICY_TO_RTP_VIEW["head_camera"]],
            "prompt": prompt,
        }

    def publish_quaternion_action(self, action: np.ndarray) -> None:
        """Set one absolute bimanual ``left(8D) + right(8D)`` target.

        Each arm is ``xyz + quaternion xyzw + gripper trigger``. When
        interpolation is enabled, a timer ramps from the last Pose actually
        published to this target over one control period.
        """
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (16,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Expected a finite 16D quaternion action, got shape {value.shape}")

        target = value.astype(np.float64, copy=True)
        for arm, start in (("left", 0), ("right", 8)):
            quaternion = target[start + 3 : start + 7]
            quaternion_norm = float(np.linalg.norm(quaternion))
            if quaternion_norm < 1e-7:
                raise ValueError(f"{arm} target quaternion has near-zero norm")
            target[start + 3 : start + 7] = quaternion / quaternion_norm
            target[start + 7] = np.clip(target[start + 7], 0.0, 1.0)

        if not self._interpolate:
            with self._cmd_lock:
                self._align_target_quaternions(target)
                self._have_target = True
                self._target_generation += 1
                generation = self._target_generation
            self._publish_command(target, generation)
            return

        # Read the measured starting pose before taking _cmd_lock. The publish
        # completion path takes _lock before _cmd_lock, so nesting these locks
        # in the opposite order here could deadlock on the first command.
        measured_start = None
        with self._cmd_lock:
            need_measured_start = self._last_cmd_action is None
        if need_measured_start:
            measured_start = self._current_quaternion_action()

        now = time.monotonic()
        with self._cmd_lock:
            if self._last_cmd_action is None:
                self._last_cmd_action = measured_start if measured_start is not None else target.copy()
            self._align_target_quaternions(target)
            self._ramp_from = self._last_cmd_action.copy()
            self._ramp_to = target
            self._ramp_t0 = now
            self._have_target = True
            self._target_generation += 1

    def _current_quaternion_action(self) -> np.ndarray | None:
        """Return current measured Poses plus the last commanded gripper values."""
        with self._lock:
            if any(arm not in self._latest_poses for arm in ("left", "right")):
                return None
            poses = {arm: _pose_to_array(self._latest_poses[arm]) for arm in ("left", "right")}
            triggers = dict(self._last_gripper_triggers)

        action = np.empty(16, dtype=np.float64)
        for arm, start in (("left", 0), ("right", 8)):
            action[start : start + 7] = poses[arm]
            quaternion = action[start + 3 : start + 7]
            action[start + 3 : start + 7] = quaternion / np.linalg.norm(quaternion)
            action[start + 7] = triggers[arm]
        return action

    def _align_target_quaternions(self, target: np.ndarray) -> None:
        """Choose quaternion signs continuously relative to the last command."""
        if self._last_cmd_action is None:
            return
        for start in (3, 11):
            if float(np.dot(self._last_cmd_action[start : start + 4], target[start : start + 4])) < 0.0:
                target[start : start + 4] *= -1.0

    def _interp_publish(self) -> None:
        with self._cmd_lock:
            if not self._have_target or self._ramp_from is None or self._ramp_to is None or self._ramp_t0 is None:
                return
            alpha = float(np.clip((time.monotonic() - self._ramp_t0) / self._ctrl_period, 0.0, 1.0))
            command = self._ramp_from + alpha * (self._ramp_to - self._ramp_from)
            for start in (3, 11):
                command[start : start + 4] = _quaternion_nlerp(
                    self._ramp_from[start : start + 4],
                    self._ramp_to[start : start + 4],
                    alpha,
                )
            generation = self._target_generation
        self._publish_command(command, generation)

    def _publish_command(self, command: np.ndarray, generation: int) -> None:
        """Publish one command unless it was superseded or output was disabled."""
        messages: dict[str, Pose] = {}
        triggers: dict[str, float] = {}
        for arm, start in (("left", 0), ("right", 8)):
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, command[start : start + 3])
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(
                float, command[start + 3 : start + 7]
            )
            messages[arm] = pose
            triggers[arm] = float(np.clip(command[start + 7], 0.0, 1.0))

        with self._publish_lock:
            with self._cmd_lock:
                if not self._have_target or generation != self._target_generation:
                    return
                first_command = not self._output_enabled
                self._output_enabled = True
                if first_command:
                    self._enable_counter = 0
                else:
                    self._enable_counter += 1
                publish_enable = first_command or self._enable_counter == 4
                if publish_enable:
                    self._enable_counter = 0

            if publish_enable:
                # Match teleavatar_v2: heartbeat every four command frames,
                # with an immediate enable before the first target.
                self._enable_publisher.publish(Float32(data=1.0))
            for arm in ("left", "right"):
                self._pose_publishers[arm].publish(messages[arm])
                self._gripper_publishers[arm].publish(Float32(data=triggers[arm]))

            published_at = time.monotonic()
            with self._lock:
                self._last_gripper_triggers.update(triggers)
            with self._cmd_lock:
                self._last_cmd_action = command.copy()
                if self._have_target and self._ramp_to is not None and generation != self._target_generation:
                    # A target arrived while this command was in the DDS
                    # publish calls. Rebase that ramp on the command which
                    # actually reached the publisher.
                    self._align_target_quaternions(self._ramp_to)
                    self._ramp_from = command.copy()
                    self._ramp_t0 = published_at

    def disable_output(self) -> None:
        published_disable = False
        with self._publish_lock:
            with self._cmd_lock:
                self._have_target = False
                self._target_generation += 1
                was_enabled = self._output_enabled
                self._output_enabled = False
                self._enable_counter = 0
                self._ramp_from = None
                self._ramp_to = None
                self._ramp_t0 = None
                self._last_cmd_action = None
            if was_enabled and rclpy.ok():
                self._enable_publisher.publish(Float32(data=0.0))
                published_disable = True
        if published_disable:
            self.get_logger().warning("Published /api/fsm/enable=0; robot output disabled")

    def destroy_node(self) -> bool:
        self._video.stop()
        return super().destroy_node()
