#!/usr/bin/env python3
"""TA2 ROS2/RTP interface for UMI relative end-effector inference."""

from __future__ import annotations

import numpy as np

from examples.teleavatar_v2.ros2_interface_endeffector import TeleavatarEndEffectorROS2Interface


class TeleavatarRelativeEndEffectorROS2Interface(TeleavatarEndEffectorROS2Interface):
    """Expose UMI's 16-D state and publish absolute actions from the model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # UMI/model convention: 0=open, 1=closed. The TA2 bring-up pose is open.
        self._commanded_gripper = {"left": 0.0, "right": 0.0}

    def get_observation(self):
        observation = super().get_observation()
        if observation is None:
            return None

        state_62d = observation["state"]
        state_16d = np.concatenate(
            (
                state_62d[48:51],
                state_62d[51:55],
                np.asarray([self._commanded_gripper["left"]], dtype=np.float32),
                state_62d[55:58],
                state_62d[58:62],
                np.asarray([self._commanded_gripper["right"]], dtype=np.float32),
            )
        ).astype(np.float32)
        return {"images": observation["images"], "state": state_16d}

    def publish_action(self, actions: np.ndarray):
        """Publish absolute [pose7, gripper] actions using UMI gripper semantics."""
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (16,):
            self.logger.error(f"Expected 16-dim action, got shape {actions.shape}")
            return

        enable_msg = self._make_float32_message(1.0)
        self.enable_pub.publish(enable_msg)

        left_pose = self._make_pose_msg("left_ee", actions[:7])
        if left_pose is not None:
            self.action_publishers["left_ee"].publish(left_pose)
        right_pose = self._make_pose_msg("right_ee", actions[8:15])
        if right_pose is not None:
            self.action_publishers["right_ee"].publish(right_pose)

        left_gripper = float(np.clip(actions[7], 0.0, 1.0))
        right_gripper = float(np.clip(actions[15], 0.0, 1.0))
        self.action_publishers["left_gripper"].publish(self._make_float32_message(left_gripper))
        self.action_publishers["right_gripper"].publish(self._make_float32_message(right_gripper))
        self._commanded_gripper.update(left=left_gripper, right=right_gripper)

    @staticmethod
    def _make_float32_message(value: float):
        from std_msgs.msg import Float32

        message = Float32()
        message.data = value
        return message
