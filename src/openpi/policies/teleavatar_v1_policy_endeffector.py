"""Policy transforms for the V1 Teleavatar robot using the end-effector action space.

Same camera pipeline as teleavatar_v1_policy.py (upside-down side-by-side
stereo head camera with left-eye crop, mono left/right cameras) and the same
v1 asymmetric gripper effort<->normalized mapping, but the model state/actions
are the end-effector poses (indices 48-61 of the 62-dim vector) instead of
joint positions. For the v2 robot's end-effector variant use
teleavatar_v2_policy_endeffector.py instead.
"""

import dataclasses
import logging

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.teleavatar_v1_policy import (
    _extract_left_head_view,
    _left_gripper_effort_to_normalized,
    _left_gripper_normalized_to_effort,
    _parse_image,
    _right_gripper_effort_to_normalized,
    _right_gripper_normalized_to_effort,
)

logger = logging.getLogger(__name__)


def make_teleavatar_endeffector_example() -> dict:
    """Creates a random input example for the Teleavatar policy with end-effector representation."""
    return {
        "observation/state": np.random.rand(62),  # 62-dim state with end-effector poses
        "observation/images/left_color": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation/images/right_color": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation/images/head_camera": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "actions": np.random.rand(62),  # 62-dim actions with end-effector poses
        "prompt": "pick a cube and place it on another cube",
    }


@dataclasses.dataclass(frozen=True)
class TeleavatarEndEffectorInputs(transforms.DataTransformFn):
    """
    Converts inputs to the model format for Teleavatar robot using end-effector representation.

    **Input format (62-dim observation/state from LeRobot dataset):**
    Layout: [joint_positions(16), joint_velocities(16), joint_efforts(16),
             left_ee_pose(7), right_ee_pose(7)]
    - Indices 0-15: Joint positions (7 left arm, 1 left gripper, 7 right arm, 1 right gripper)
    - Indices 16-31: Joint velocities (same layout)
    - Indices 32-47: Joint efforts (same layout)
    - Indices 48-54: Left end-effector pose (x, y, z, qx, qy, qz, qw) - CURRENT pose
    - Indices 55-61: Right end-effector pose (x, y, z, qx, qy, qz, qw) - CURRENT pose

    **Model state format (14-dim):**
    [left_ee_pose(7), right_ee_pose(7)] — from input[48:55] and input[55:62].

    **Model actions (16-dim, training only):**
    [left_ee_target_pose(7), left_gripper(1), right_ee_target_pose(7), right_gripper(1)]
    with gripper efforts converted to the normalized [0, 1] controller range
    (same asymmetric v1 mapping as teleavatar_v1_policy.py).

    **Cameras (v1 robot):** the head camera is side-by-side stereo — the left
    eye is cropped out (rotated 180° first iff rotate_head_camera). The
    left/right cameras are mono and pass through the shape guard untouched.
    """
    model_type: _model.ModelType
    # Whether to rotate 180° before cropping the head frame. Property of the
    # source data, not of train/inference — set it identically for both. The
    # 2:1-width guard inside _extract_left_head_view already no-ops for
    # already-cropped (square) frames, so this only matters for raw stereo.
    #   True  → upside-down raw stereo (camera mounted upside-down, e.g. the
    #           officially released robot — its training data is upside-down too)
    #   False → frame already right-side-up before this transform
    rotate_head_camera: bool = False

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) format
        # LeRobot stores as float32 (C,H,W) during training, but runtime sends uint8 (H,W,C)
        left_color = _parse_image(data["observation/images/left_color"])
        right_color = _parse_image(data["observation/images/right_color"])
        head_color = _parse_image(data["observation/images/head_camera"])
        # Crop the left eye from the side-by-side stereo head frame, rotating
        # 180° iff the configured source orientation says so. The width-check
        # guard inside _extract_left_head_view makes this a no-op when the head
        # frame already arrives square (cropped by ros2_interface).
        head_color = _extract_left_head_view(
            head_color, rotate=self.rotate_head_camera
        )

        # Extract 14-dim end-effector state from extended observation
        # Using end-effector representation instead of joint angles
        state_14d = np.concatenate([
            data["observation/state"][48:55],  # Left arm end-effector (x,y,z,qx,qy,qz,qw)
            data["observation/state"][55:62],  # Right arm end-effector (x,y,z,qx,qy,qz,qw)
        ], axis=0)

        # Create inputs dict. Do not change the keys in the dict below.
        # Pi0 models support three image inputs: one third-person view and two wrist views.
        # Map teleavatar cameras to the expected model inputs.
        inputs = {
            "state": state_14d,
            "image": {
                "base_0_rgb": head_color,
                "left_wrist_0_rgb": left_color,
                "right_wrist_0_rgb": right_color,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Extract 16-dim TARGET actions from 62-dim action space during training
        # Actions are only available during training, not during inference
        if "action" in data:
            # data["action"] has shape [action_horizon, 62]
            # Layout: [joint_positions(16), joint_velocities(16), joint_efforts(16),
            #          left_ee_target_pose(7), right_ee_target_pose(7)]
            action_data = data["action"]

            # Extract TARGET end-effector poses (indices 48-61)
            selected_actions = np.concatenate([
                action_data[:, 48:55],  # Left arm TARGET end-effector pose
                action_data[:, 39:40],  # Left gripper effort (index 39 = 32+7)
                action_data[:, 55:62],  # Right arm TARGET end-effector pose
                action_data[:, 47:48],  # Right gripper effort (index 47 = 32+15)
            ], axis=1)  # Concatenate along action dimension

            # Convert raw gripper efforts to the normalized [0, 1] controller
            # range (same asymmetric v1 mapping as teleavatar_v1_policy.py).
            # This runs before the dataset norm-stats normalization in
            # model_transforms, so norm stats must be recomputed after enabling
            # this (scripts/compute_norm_stats.py). Action layout is
            # [left_ee(7), left_gripper(1), right_ee(7), right_gripper(1)].
            selected_actions[:, 7] = _left_gripper_effort_to_normalized(selected_actions[:, 7])
            selected_actions[:, 15] = _right_gripper_effort_to_normalized(selected_actions[:, 15])

            inputs["actions"] = selected_actions

        # Pass the prompt (aka language instruction) to the model. During
        # training this should be filled by PromptFromLeRobotTask + the
        # "prompt": "prompt" entry in the repack structure (see
        # LeRobotTeleavatarV1EndEffectorDataConfig.create). The fallback below
        # only kicks in for inference callers that don't supply a prompt; if it
        # ever triggers during training, every sample shares one fixed string
        # and the language channel is dead — make that visible.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        else:
            inputs["prompt"] = "perform the manipulation task"
            logger.warning(
                "TeleavatarEndEffectorInputs: no 'prompt' in sample; using "
                "hardcoded fallback. Expected during inference without a client "
                "prompt, but indicates a training-pipeline bug if seen on "
                "training data."
            )

        return inputs


@dataclasses.dataclass(frozen=True)
class TeleavatarEndEffectorOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. It is
    used for inference only.

    For teleavatar with end-effector representation, we return 16 actions:
    - End-effector TARGET pose (x,y,z,qx,qy,qz,qw) for both arms (14 values)
    - Gripper efforts for left and right grippers (2 values)

    **Why extract only 16 dimensions?**
    - Model is configured with action_dim=32 to match pre-trained pi0_base weights
    - But we only need 16 dimensions for our robot (2 arms × 7 DOF + 2 grippers)
    - The extra dimensions (16-31) are padding and should be discarded
    - This allows us to leverage pre-trained weights while adapting to our robot
    """

    def __call__(self, data: dict) -> dict:
        # Extract only the first 16 actions from model output (action_dim=32)
        # Model output has padding dimensions that we don't need
        # Output format: [left_ee_target_pose(7), left_gripper_effort(1),
        #                 right_ee_target_pose(7), right_gripper_effort(1)]
        # Copy so the in-place gripper conversion below never mutates the
        # caller's array.
        actions = np.array(data["actions"][:, :16])

        # Convert normalized [0, 1] gripper values back to effort for robot
        # execution (inverse of the effort->normalized map in
        # TeleavatarEndEffectorInputs; same asymmetric v1 mapping as
        # teleavatar_v1_policy.py).
        actions[:, 7] = _left_gripper_normalized_to_effort(actions[:, 7])
        actions[:, 15] = _right_gripper_normalized_to_effort(actions[:, 15])
        return {"actions": actions}

