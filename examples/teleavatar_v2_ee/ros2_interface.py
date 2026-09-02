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
    """Collect bimanual EE observations and publish direct platform API commands.

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
        initial_left_gripper_trigger: float = 0.0,
        initial_right_gripper_trigger: float = 0.0,
        enable_frequency: float = 20.0,
        node_name: str = "teleavatar_v2_ee_openpi_interface",
    ) -> None:
        super().__init__(node_name)
        if sensor_timeout <= 0.0:
            raise ValueError("sensor_timeout must be positive")
        if enable_frequency <= 0.0:
            raise ValueError("enable_frequency must be positive")

        self.sensor_timeout = float(sensor_timeout)
        self._lock = Lock()
        self._latest_poses: dict[str, Pose] = {}
        self._pose_timestamps: dict[str, float] = {}
        self._last_gripper_triggers = {
            "left": float(np.clip(initial_left_gripper_trigger, 0.0, 1.0)),
            "right": float(np.clip(initial_right_gripper_trigger, 0.0, 1.0)),
        }
        self._output_enabled = False

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
        self._enable_timer = self.create_timer(1.0 / enable_frequency, self._publish_enable_heartbeat)

        self.get_logger().info(
            f"Bimanual EE interface ready: poses={list(self._pose_topics.values())}, RTP=udp/{rtp_port}, "
            f"initial gripper triggers={self._last_gripper_triggers}"
        )

    def _publish_enable_heartbeat(self) -> None:
        with self._lock:
            output_enabled = self._output_enabled
        if output_enabled:
            self._enable_publisher.publish(Float32(data=1.0))

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

        images, _timestamps = self._video.get_latest_images_with_timestamps()
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
        """Publish one absolute bimanual ``left(8D) + right(8D)`` action.

        Each arm is ``xyz + quaternion xyzw + gripper trigger``. The caller
        has already converted the policy's rot6d output and interpolated the
        quaternion, so no rotation representation conversion happens here.
        """
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (16,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Expected a finite 16D quaternion action, got shape {value.shape}")

        messages: dict[str, Pose] = {}
        triggers: dict[str, float] = {}
        for arm, start in (("left", 0), ("right", 8)):
            quaternion = value[start + 3 : start + 7]
            quaternion_norm = float(np.linalg.norm(quaternion))
            if quaternion_norm < 1e-7:
                raise ValueError(f"{arm} target quaternion has near-zero norm")
            quaternion = quaternion / quaternion_norm
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, value[start : start + 3])
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, quaternion)
            messages[arm] = pose
            triggers[arm] = float(np.clip(value[start + 7], 0.0, 1.0))

        with self._lock:
            self._last_gripper_triggers.update(triggers)
            self._output_enabled = True
        # Enable before the first target so the platform does not discard the
        # initial pose while transitioning into API control.
        self._enable_publisher.publish(Float32(data=1.0))
        for arm in ("left", "right"):
            self._pose_publishers[arm].publish(messages[arm])
            self._gripper_publishers[arm].publish(Float32(data=triggers[arm]))

    def disable_output(self) -> None:
        with self._lock:
            was_enabled = self._output_enabled
            self._output_enabled = False
        if was_enabled and rclpy.ok():
            self._enable_publisher.publish(Float32(data=0.0))
            self.get_logger().warning("Published /api/fsm/enable=0; robot output disabled")

    def destroy_node(self) -> bool:
        self._video.stop()
        return super().destroy_node()
