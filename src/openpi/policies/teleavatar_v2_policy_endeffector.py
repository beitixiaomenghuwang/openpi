"""Policy transforms for the V2 Teleavatar robot using the end-effector action space.

Same camera pipeline as teleavatar_v2_policy.py (side-by-side stereo on all
three cameras, one eye cropped per camera) and the same v2 gripper
trigger<->effort curve, but the model state/actions are the end-effector poses
(indices 48-61 of the 62-dim vector) instead of joint positions. For the v1
robot's end-effector variant use teleavatar_policy_endeffector.py instead.
"""

import dataclasses
import logging

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.teleavatar_v2_policy import (
    _extract_stereo_view,
    _gripper_effort_to_trigger,
    _gripper_trigger_to_effort,
    _parse_image,
)

logger = logging.getLogger(__name__)


def make_teleavatar_v2_endeffector_example() -> dict:
    """Creates a random input example for the Teleavatar v2 end-effector policy."""
    return {
        "observation/state": np.random.rand(62),
        "observation/images/left_color": np.random.randint(256, size=(800, 2560, 3), dtype=np.uint8),
        "observation/images/right_color": np.random.randint(256, size=(800, 2560, 3), dtype=np.uint8),
        "observation/images/head_camera": np.random.randint(256, size=(1920, 3840, 3), dtype=np.uint8),
        "actions": np.random.rand(62),
        "prompt": "pick a cube and place it on another cube",
    }


@dataclasses.dataclass(frozen=True)
class TeleavatarEndEffectorInputs(transforms.DataTransformFn):
    """
    Converts inputs to the model format for the Teleavatar v2 robot using the
    end-effector representation.

    **Input format (observation/state from LeRobot dataset):**
    Layout: [positions(16), velocities(16), efforts(16), ee_poses(14)]
    (62-dim; 72-dim datasets append chassis dims after index 61 — the indices
    used below are identical):
    - Indices 0-15: Joint positions (7 left arm, 1 left gripper, 7 right arm, 1 right gripper)
    - Indices 16-31: Joint velocities (same layout)
    - Indices 32-47: Joint efforts (same layout)
    - Indices 48-54: Left end-effector pose (x, y, z, qx, qy, qz, qw) — CURRENT pose
    - Indices 55-61: Right end-effector pose (x, y, z, qx, qy, qz, qw) — CURRENT pose

    **Model state format (14-dim):**
    [left_ee_pose(7), right_ee_pose(7)] — from input[48:55] and input[55:62].

    **Model actions (16-dim, training only):**
    [left_ee_target_pose(7), left_gripper(1), right_ee_target_pose(7), right_gripper(1)]
    with gripper efforts converted to the normalized [0, 1] trigger range
    (same v2 curve for both arms).

    **Cameras (v2 robot):** all three streams are side-by-side stereo; one eye
    is cropped out per camera (see __call__). The shape guard in
    _extract_stereo_view leaves already-cropped frames untouched.
    """
    model_type: _model.ModelType
    # Whether to rotate 180° before cropping the head frame. Property of the
    # source data, not of train/inference — set it identically for both.
    #   True  → upside-down raw stereo (e.g. v1-style upside-down head mount)
    #   False → frame already right-side-up before this transform (v2 robot)
    rotate_head_camera: bool = False

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) format
        # LeRobot stores as float32 (C,H,W) during training, but runtime sends uint8 (H,W,C)
        left_color = _parse_image(data["observation/images/left_color"])
        right_color = _parse_image(data["observation/images/right_color"])
        head_color = _parse_image(data["observation/images/head_camera"])
        # Crop one eye from each side-by-side stereo frame. The head keeps its
        # left eye (rotated 180° first iff the configured source orientation
        # says so). The left/right cameras keep their INNER eye — right eye of
        # the left camera, left eye of the right camera — so both look at the
        # middle of the desktop workspace. The width guard inside
        # _extract_stereo_view makes all three no-ops when frames already
        # arrive cropped (by ros2_interface).
        head_color = _extract_stereo_view(
            head_color, "left", rotate=self.rotate_head_camera
        )
        left_color = _extract_stereo_view(left_color, "right")
        right_color = _extract_stereo_view(right_color, "left")

        # Extract the 14-dim end-effector state from the observation vector.
        # Input layout: [positions(0-15), velocities(16-31), efforts(32-47), ee_pose(48-61)]
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

        # Extract 16-dim TARGET actions from the dataset action vector during
        # training. Actions are only available during training, not during inference.
        if "action" in data:
            # data["action"] has shape [action_horizon, 62] (or [action_horizon, 72])
            # Layout: [positions(0-15), velocities(16-31), efforts(32-47), ee_pose(48-61)]
            action_data = data["action"]

            selected_actions = np.concatenate([
                action_data[:, 48:55],  # Left arm TARGET end-effector pose
                action_data[:, 39:40],  # Left gripper effort (index 39 = 32+7)
                action_data[:, 55:62],  # Right arm TARGET end-effector pose
                action_data[:, 47:48],  # Right gripper effort (index 47 = 32+15)
            ], axis=1)  # Concatenate along action dimension

            # Convert raw gripper efforts (Nm) to the normalized [0, 1]
            # trigger range (same curve for both arms on v2). This runs before
            # the dataset norm-stats normalization in model_transforms, so norm
            # stats must be recomputed after changing this mapping
            # (scripts/compute_norm_stats.py). Action layout is
            # [left_ee(7), left_gripper(1), right_ee(7), right_gripper(1)].
            selected_actions[:, 7] = _gripper_effort_to_trigger(selected_actions[:, 7])
            selected_actions[:, 15] = _gripper_effort_to_trigger(selected_actions[:, 15])

            inputs["actions"] = selected_actions

        # Pass the prompt (aka language instruction) to the model. During
        # training this should be filled by PromptFromLeRobotTask + the
        # "prompt": "prompt" entry in the repack structure (see
        # LeRobotTeleavatarV2EndEffectorDataConfig.create). The fallback below
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

    For teleavatar v2 with end-effector representation, we return 16 actions:
    - End-effector TARGET pose (x,y,z,qx,qy,qz,qw) for both arms (14 values)
    - Gripper efforts (Nm) for left and right grippers (2 values)
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 16 actions; the model may output more
        # dimensions due to padding (action_dim=32).
        # Copy so the in-place gripper conversion below never mutates the
        # caller's array.
        actions = np.array(data["actions"][:, :16])

        # Convert normalized [0, 1] trigger values back to effort (Nm) for
        # robot execution (inverse of the effort→trigger map in
        # TeleavatarEndEffectorInputs; same curve for both arms on v2).
        actions[:, 7] = _gripper_trigger_to_effort(actions[:, 7])
        actions[:, 15] = _gripper_trigger_to_effort(actions[:, 15])
        return {"actions": actions}
