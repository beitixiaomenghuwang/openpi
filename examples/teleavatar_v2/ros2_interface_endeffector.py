#!/usr/bin/env python3
"""
Sensor/actuator interface for the Teleavatar v2 robot with end-effector control.
Joint states and end-effector poses go over ROS2; camera images arrive as a
single RTP/H265 composite stream decoded with GStreamer, exactly like the
joint-space deployment (see ros2_interface.py and rtp_video_interface.py).

Differences from ros2_interface.py (joint-space v2 deployment):
- Additionally subscribes to the CURRENT end-effector pose of each arm and
  appends it to the state vector (62-dim: [positions(16), velocities(16),
  efforts(16), left_ee_pose(7), right_ee_pose(7)]). The EE policy only reads
  indices 48-61; the joint block is filled from /​<arm>/joint_states anyway
  (gripper dims left 0, same as the joint deployment).
- Publishes TARGET end-effector poses (geometry_msgs/Pose) to the platform
  API topics /api/left_arm/target_pose and /api/right_arm/target_pose
  instead of joint position commands, so no joint-limit clamping happens
  here (the platform's IK owns that).
- Grippers identical to the joint deployment: the model's effort (Nm) is
  converted with the v2 piecewise trigger curve to a [0, 1] Float32 on
  /api/<side>_gripper/cmd, and /api/fsm/enable is pinged every tick.
"""

import logging
import pathlib
import sys
import time
from threading import Lock
from typing import Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

# Make the repo root importable regardless of cwd (for the package import below).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.teleavatar_v2.rtp_video_interface import RTPH265VideoInterface  # noqa: E402

# Maps the observation keys the policy expects (same as the training dataset
# keys) to the RTP split-view names. Note the RTP interface also exposes a
# "head_camera" key of its own — that one is the FULL 2720×1280 composite,
# not the head view, so the mapping below must be used instead of passing
# the RTP dict through.
_POLICY_TO_RTP_VIEW = {
    "head_camera": "head_left_eye",
    "left_color": "left_wrist_right_eye",
    "right_color": "right_wrist_left_eye",
}


def _gripper_effort_to_trigger(effort: float) -> float:
    """Model gripper effort (Nm) → platform [0, 1] trigger value.

    Inverse of the v2 piecewise trigger→effort curve (same for both arms),
    identical to the conversion in ros2_interface.publish_action. Clipped
    because the platform expects a 0~1 trigger.
    """
    if effort > 0:
        trigger = 0.10 * (1.0 - effort / 2.0)
    else:
        trigger = 0.10 - effort * 0.90 / 1.6
    return float(np.clip(trigger, 0.0, 1.0))


class TeleavatarEndEffectorROS2Interface(Node):
    """Thread-safe interface for Teleavatar v2 sensors (ROS2 joints + EE poses
    + RTP video) and end-effector actuation."""

    def __init__(
        self,
        node_name: str = "teleavatar_endeffector_openpi_interface",
        rtp_port: int = 8890,
        rtp_payload: int = 96,
        sensor_timeout: float = 1.0,
        left_ee_pose_topic: str = '/left_arm/current_ee_pose',
        right_ee_pose_topic: str = '/right_arm/current_ee_pose',
    ):
        super().__init__(node_name)

        self.logger = self.get_logger()
        self.lock = Lock()
        # Sensor data older than this (seconds) is treated as dead (video
        # ~45 fps, joint states ~100 Hz): get_observation returns None
        # instead of the frozen last sample.
        self.sensor_timeout = sensor_timeout
        self.ee_pose_topics = {
            'left_ee': left_ee_pose_topic,
            'right_ee': right_ee_pose_topic,
        }

        # Storage for latest joint/EE data (images live in the RTP interface)
        self.latest_joint_states: Dict[str, JointState] = {}
        self.joint_timestamps: Dict[str, float] = {}
        self.latest_ee_poses: Dict[str, Pose] = {}
        self.ee_pose_timestamps: Dict[str, float] = {}

        # Camera images arrive over RTP, not ROS2 (see module docstring). The
        # interface logs its own fps/decode-latency stats periodically.
        self._video = RTPH265VideoInterface(port=rtp_port, payload=rtp_payload)
        self._video.start()

        # Setup subscribers and publishers
        self._setup_subscribers()
        self._setup_publishers()

        self.logger.info("TeleavatarEndEffectorROS2Interface initialized (waiting for sensor data in background)")

    def _setup_subscribers(self):
        """Setup ROS2 subscribers for joint states and current EE poses."""
        self.create_subscription(
            JointState,
            '/left_arm/joint_states',
            lambda msg: self._joint_state_callback(msg, 'left_arm'),
            10
        )
        self.create_subscription(
            JointState,
            '/right_arm/joint_states',
            lambda msg: self._joint_state_callback(msg, 'right_arm'),
            10
        )

        # CURRENT end-effector poses (input to the model, state indices 48-61)
        self.create_subscription(
            Pose,
            self.ee_pose_topics['left_ee'],
            lambda msg: self._ee_pose_callback(msg, 'left_ee'),
            10
        )
        self.create_subscription(
            Pose,
            self.ee_pose_topics['right_ee'],
            lambda msg: self._ee_pose_callback(msg, 'right_ee'),
            10
        )

        self.logger.info("ROS2 subscribers initialized (joint states + current end-effector poses)")

    def _setup_publishers(self):
        """Setup ROS2 publishers for end-effector action commands.

        TARGET EE poses go to the platform API topics (the platform computes
        and clamps the joint commands itself). Grippers and FSM enable use the
        same /api topics as the joint-space deployment.
        """
        self.action_publishers = {
            'left_ee': self.create_publisher(Pose, '/api/left_arm/target_pose', 10),
            'right_ee': self.create_publisher(Pose, '/api/right_arm/target_pose', 10),
            'left_gripper': self.create_publisher(Float32, '/api/left_gripper/cmd', 10),
            'right_gripper': self.create_publisher(Float32, '/api/right_gripper/cmd', 10),
        }
        self.enable_pub = self.create_publisher(Float32, '/api/fsm/enable', 10)
        self.logger.info("ROS2 publishers initialized (target end-effector pose control)")

    def _joint_state_callback(self, msg: JointState, joint_group: str):
        """Callback for joint state messages."""
        with self.lock:
            self.latest_joint_states[joint_group] = msg
            self.joint_timestamps[joint_group] = time.time()

    def _ee_pose_callback(self, msg: Pose, ee_name: str):
        """Callback for current end-effector pose messages."""
        with self.lock:
            self.latest_ee_poses[ee_name] = msg
            self.ee_pose_timestamps[ee_name] = time.time()

    def destroy_node(self):
        """Stop the RTP video pipeline before tearing down the ROS2 node."""
        try:
            self._video.stop()
        finally:
            super().destroy_node()

    def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
        """Wait for initial sensor data (first RTP video frame + joint states
        + current EE poses).

        NOTE: This should be called AFTER the ROS2 node starts spinning,
        otherwise the callbacks will never be triggered! (The RTP video
        thread runs independently of the ROS2 executor.)

        Returns:
            True if all data received, False if timeout
        """
        required_joints = ['left_arm', 'right_arm']
        required_ee_poses = ['left_ee', 'right_ee']

        start_time = time.time()
        self.logger.info("Waiting for initial sensor data (including end-effector poses)...")

        last_status_time = start_time
        while time.time() - start_time < timeout:
            video_ready = self._video.has_initial_frame()
            with self.lock:
                joints_ready = all(joint in self.latest_joint_states for joint in required_joints)
                ee_poses_ready = all(ee in self.latest_ee_poses for ee in required_ee_poses)
                have_joints = [joint for joint in required_joints if joint in self.latest_joint_states]
                have_ee_poses = [ee for ee in required_ee_poses if ee in self.latest_ee_poses]

            if video_ready and joints_ready and ee_poses_ready:
                self.logger.info("✓ All sensor data received!")
                return True

            # Log progress every 2 seconds
            if time.time() - last_status_time > 2.0:
                self.logger.info(
                    f"  Progress: video={video_ready}, joints={have_joints}, ee_poses={have_ee_poses}"
                )
                last_status_time = time.time()

            time.sleep(0.1)

        # Timeout - log what's missing
        with self.lock:
            missing_joints = [joint for joint in required_joints if joint not in self.latest_joint_states]
            missing_ee_poses = [ee for ee in required_ee_poses if ee not in self.latest_ee_poses]

        self.logger.error(
            f"✗ Timeout waiting for sensor data after {timeout}s. "
            f"Missing: video={not self._video.has_initial_frame()} (RTP port {self._video.port}), "
            f"joints={missing_joints}, "
            f"ee_poses={missing_ee_poses} (topics {list(self.ee_pose_topics.values())})"
        )
        return False

    def get_observation(self) -> Optional[Dict]:
        """Get current observation from all sensors.

        Returns:
            Dictionary with 'images' and 'state' keys, or None if any sensor
            is missing or older than sensor_timeout. Callers must treat None
            as "do not act" — never fall back to a previous observation.
        """
        now = time.time()
        dead: list = []

        if self._video.stream_ended():
            dead.append("video: RTP pipeline stopped (EOS/error)")

        # Split views from the RTP stream (already copies; no shared buffers).
        rtp_images, rtp_stamps = self._video.get_latest_images_with_timestamps()
        for view in _POLICY_TO_RTP_VIEW.values():
            stamp = rtp_stamps.get(view)
            if view not in rtp_images or stamp is None:
                dead.append(f"video:{view}: not received")
            elif now - stamp > self.sensor_timeout:
                dead.append(f"video:{view}: {now - stamp:.1f}s stale")

        with self.lock:
            left_arm = self.latest_joint_states.get('left_arm')
            right_arm = self.latest_joint_states.get('right_arm')
            joint_stamps = dict(self.joint_timestamps)
            left_ee_pose = self.latest_ee_poses.get('left_ee')
            right_ee_pose = self.latest_ee_poses.get('right_ee')
            ee_stamps = dict(self.ee_pose_timestamps)

        for joint_group in ('left_arm', 'right_arm'):
            stamp = joint_stamps.get(joint_group)
            if stamp is None:
                dead.append(f"joints:{joint_group}: not received")
            elif now - stamp > self.sensor_timeout:
                dead.append(f"joints:{joint_group}: {now - stamp:.1f}s stale")

        for ee_name in ('left_ee', 'right_ee'):
            stamp = ee_stamps.get(ee_name)
            if stamp is None:
                dead.append(f"ee_pose:{ee_name}: not received")
            elif now - stamp > self.sensor_timeout:
                dead.append(f"ee_pose:{ee_name}: {now - stamp:.1f}s stale")

        if dead:
            self.logger.error(
                "Observation unavailable — dead sensors: " + "; ".join(dead),
                throttle_duration_sec=1.0,
            )
            return None

        # Build 62-dimensional state vector
        # Layout: [positions(16), velocities(16), efforts(16),
        #          left_ee_pose(7), right_ee_pose(7)]
        # The EE policy only reads indices 48-61; the joint block is filled
        # anyway for debugging/consistency (gripper dims left 0, same as the
        # joint-space deployment — v2 publishes no gripper joint states).
        state_62d = np.zeros(62, dtype=np.float32)

        # Positions (indices 0-15)
        state_62d[0:7] = self._extract_joint_field(left_arm, 'position', 7)
        state_62d[8:15] = self._extract_joint_field(right_arm, 'position', 7)

        # Velocities (indices 16-31)
        state_62d[16:23] = self._extract_joint_field(left_arm, 'velocity', 7)
        state_62d[24:31] = self._extract_joint_field(right_arm, 'velocity', 7)

        # Efforts (indices 32-47)
        state_62d[32:39] = self._extract_joint_field(left_arm, 'effort', 7)
        state_62d[40:47] = self._extract_joint_field(right_arm, 'effort', 7)

        # End-effector poses (indices 48-61): (x, y, z, qx, qy, qz, qw) per arm
        state_62d[48:55] = self._pose_to_array(left_ee_pose)
        state_62d[55:62] = self._pose_to_array(right_ee_pose)

        return {
            'images': {policy_key: rtp_images[view] for policy_key, view in _POLICY_TO_RTP_VIEW.items()},
            'state': state_62d,
        }

    @staticmethod
    def _pose_to_array(pose: Pose) -> np.ndarray:
        """Convert geometry_msgs/Pose to [x, y, z, qx, qy, qz, qw]."""
        return np.array([
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
        ], dtype=np.float32)

    def _extract_joint_field(self, msg: JointState, field: str, num_joints: int) -> np.ndarray:
        """Extract joint data field (position/velocity/effort) from JointState message."""
        data = getattr(msg, field, [])

        if len(data) >= num_joints:
            return np.array(data[:num_joints], dtype=np.float32)
        else:
            # Pad with zeros if not enough data
            result = np.zeros(num_joints, dtype=np.float32)
            result[:len(data)] = data
            return result

    def _make_pose_msg(self, arm: str, pose7: np.ndarray) -> Optional[Pose]:
        """Build a Pose message from [x, y, z, qx, qy, qz, qw], normalizing the
        quaternion. The model output is only approximately unit-norm; the IK
        node expects a valid rotation. Returns None (and logs) for a
        degenerate quaternion — the caller must skip publishing that arm."""
        quat = np.asarray(pose7[3:7], dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if norm < 1e-3:
            self.logger.error(f"{arm}: degenerate target quaternion {quat} — command dropped")
            return None
        if abs(norm - 1.0) > 0.05:
            self.logger.warning(
                f"{arm}: target quaternion norm {norm:.3f} far from 1 — check the policy output",
                throttle_duration_sec=2.0,
            )
        quat = quat / norm

        msg = Pose()
        msg.position.x = float(pose7[0])
        msg.position.y = float(pose7[1])
        msg.position.z = float(pose7[2])
        msg.orientation.x = float(quat[0])
        msg.orientation.y = float(quat[1])
        msg.orientation.z = float(quat[2])
        msg.orientation.w = float(quat[3])
        return msg

    def publish_action(self, actions: np.ndarray):
        """Publish 16-dimensional TARGET action to ROS topics.

        Publishes target end-effector poses to the platform API topics
        (/api/left_arm/target_pose, /api/right_arm/target_pose) and gripper
        trigger values to /api/<side>_gripper/cmd (model effort (Nm) → v2
        trigger curve, clipped to [0, 1]).

        Args:
            actions: 16-dim array [left_ee_target_pose(7), left_gripper_effort(1),
                                   right_ee_target_pose(7), right_gripper_effort(1)]
        """
        if actions.shape != (16,):
            self.logger.error(f"Expected 16-dim action, got shape {actions.shape}")
            return

        # Enable FSM
        enable_msg = Float32()
        enable_msg.data = 1.0
        self.enable_pub.publish(enable_msg)

        # Left end-effector TARGET pose (predicted by the model)
        left_ee_msg = self._make_pose_msg('left_ee', actions[0:7])
        if left_ee_msg is not None:
            self.action_publishers['left_ee'].publish(left_ee_msg)

        # Left gripper: effort (Nm) → [0, 1] trigger
        left_gripper_msg = Float32()
        left_gripper_msg.data = _gripper_effort_to_trigger(float(actions[7]))
        self.action_publishers['left_gripper'].publish(left_gripper_msg)

        # Right end-effector TARGET pose (predicted by the model)
        right_ee_msg = self._make_pose_msg('right_ee', actions[8:15])
        if right_ee_msg is not None:
            self.action_publishers['right_ee'].publish(right_ee_msg)

        # Right gripper: same effort → trigger conversion as the left.
        right_gripper_msg = Float32()
        right_gripper_msg.data = _gripper_effort_to_trigger(float(actions[15]))
        self.action_publishers['right_gripper'].publish(right_gripper_msg)
