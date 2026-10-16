"""Policy transforms for the YUBI UMI Arena dataset (airoa-org/yubi-corl2026-umi-arena).

The model predicts action chunks relative to the current hand_root pose; ``YubiUMIOutputs``
turns them into the UMI Arena publish vector (per-step body-frame pose deltas + absolute
gripper angles). Keys are the dataset's native feature names, which are also the keys the
evaluation harness sends, so the same transforms serve training and the submission.

Only numpy, einops and ``openpi.transforms`` are used: the evaluation image runs upstream
openpi, so this file has to be copyable next to the submission's ``policy.py``.
"""

import dataclasses
import re

import einops
import numpy as np

from openpi import transforms

LEFT_IMAGE = "observation.image.left"
RIGHT_IMAGE = "observation.image.right"
# Right hand_root pose expressed in the left hand_root frame, [x, y, z, qx, qy, qz, qw].
INTER_HAND_POSE = "observation.pose.left_hand_root_to_right_hand_root.absolute"
# Gripper motor angles [left, right] in rad.
JOINT_STATES = "observation.joint_states"
# Training-only labels: absolute hand_root poses in the table frame, and the next-frame gripper
# angles (action.joint_states[t] == observation.joint_states[t + 1]).
LEFT_HAND_POSE = "observation.pose.left_hand_root.absolute"
RIGHT_HAND_POSE = "observation.pose.right_hand_root.absolute"
JOINT_ACTIONS = "action.joint_states"
ACTION_SEQUENCE_KEYS = (LEFT_HAND_POSE, RIGHT_HAND_POSE, JOINT_ACTIONS)
# Frame t anchors the chunk, so the hand poses are loaded for t..t+H (one more than the horizon).
ACTION_SEQUENCE_EXTRA_STEPS = 1

# Model state: inter-hand [xyz(3), rotation-6D(6)] + gripper angles (2).
YUBI_STATE_DIM = 11
# Model action per arm: [relative xyz(3), relative rotation-6D(6), gripper(1)], left arm first.
YUBI_ARM_ACTION_DIM = 10
YUBI_ACTION_DIM = 2 * YUBI_ARM_ACTION_DIM
# Publish vector: [left delta(7), right delta(7), left gripper, right gripper].
YUBI_PUBLISH_DIM = 16
# Finger joint limits from yubi_description/urdf/yubi_hand.urdf.xacro (airoa-org/yubi-sw).
GRIPPER_LIMITS = (0.0, 0.94)

_PROMPT_NUMBERING = re.compile(r"^\d+\.\s*")


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def normalize_prompt(prompt: str) -> str:
    """Drop the step numbering ("1.", "7. ") and trailing period some dataset instructions carry."""
    prompt = _PROMPT_NUMBERING.sub("", str(prompt).strip())
    return prompt.rstrip(".").strip()


def _quat_xyzw_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions to rotation matrices."""
    quaternion = np.array(quaternion, dtype=np.float32, copy=True)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
        # An unbatched quaternion makes x/y/z/w 0-d, which numpy's legacy promotion
        # upcasts to float64; pin the dtype so the transform output does not depend on rank.
        dtype=np.float32,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Recover a row-major rotation matrix from the first two rows."""
    rotation_6d = np.asarray(rotation_6d, dtype=np.float32)
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.maximum(np.linalg.norm(a2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def _rotation_matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to unit xyzw quaternions with qw >= 0."""
    rotation = np.asarray(rotation, dtype=np.float32)
    original_shape = rotation.shape[:-2]
    rotation = rotation.reshape(-1, 3, 3)
    result = np.empty((rotation.shape[0], 4), dtype=np.float32)
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    positive = trace > 0
    s = np.sqrt(np.maximum(trace[positive] + 1.0, 1e-8)) * 2
    result[positive, 3] = 0.25 * s
    result[positive, 0] = (rotation[positive, 2, 1] - rotation[positive, 1, 2]) / s
    result[positive, 1] = (rotation[positive, 0, 2] - rotation[positive, 2, 0]) / s
    result[positive, 2] = (rotation[positive, 1, 0] - rotation[positive, 0, 1]) / s

    remaining = ~positive
    diagonal = np.argmax(np.diagonal(rotation[remaining], axis1=-2, axis2=-1), axis=-1)
    for index in range(3):
        selected = remaining.copy()
        selected[remaining] &= diagonal == index
        if index == 0:
            s = (
                np.sqrt(
                    np.maximum(
                        1.0 + rotation[selected, 0, 0] - rotation[selected, 1, 1] - rotation[selected, 2, 2], 1e-8
                    )
                )
                * 2
            )
            result[selected, 0] = 0.25 * s
            result[selected, 1] = (rotation[selected, 0, 1] + rotation[selected, 1, 0]) / s
            result[selected, 2] = (rotation[selected, 0, 2] + rotation[selected, 2, 0]) / s
            result[selected, 3] = (rotation[selected, 2, 1] - rotation[selected, 1, 2]) / s
        elif index == 1:
            s = (
                np.sqrt(
                    np.maximum(
                        1.0 + rotation[selected, 1, 1] - rotation[selected, 0, 0] - rotation[selected, 2, 2], 1e-8
                    )
                )
                * 2
            )
            result[selected, 0] = (rotation[selected, 0, 1] + rotation[selected, 1, 0]) / s
            result[selected, 1] = 0.25 * s
            result[selected, 2] = (rotation[selected, 1, 2] + rotation[selected, 2, 1]) / s
            result[selected, 3] = (rotation[selected, 0, 2] - rotation[selected, 2, 0]) / s
        else:
            s = (
                np.sqrt(
                    np.maximum(
                        1.0 + rotation[selected, 2, 2] - rotation[selected, 0, 0] - rotation[selected, 1, 1], 1e-8
                    )
                )
                * 2
            )
            result[selected, 0] = (rotation[selected, 0, 2] + rotation[selected, 2, 0]) / s
            result[selected, 1] = (rotation[selected, 1, 2] + rotation[selected, 2, 1]) / s
            result[selected, 2] = 0.25 * s
            result[selected, 3] = (rotation[selected, 1, 0] - rotation[selected, 0, 1]) / s
    result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-8)
    # The publish contract treats q and -q alike, but a fixed hemisphere keeps the output deterministic.
    result *= np.where(result[:, 3:4] < 0, -1.0, 1.0).astype(np.float32)
    return result.reshape(*original_shape, 4)


def _inter_hand_state(inter_hand_pose: np.ndarray, joint_states: np.ndarray) -> np.ndarray:
    """Build the 11-D model state. The dataset's inter-hand quaternions are not canonicalized
    (q and -q both occur), so the rotation is passed as rotation-6D, which is sign-free."""
    inter_hand_pose = np.asarray(inter_hand_pose, dtype=np.float32)
    rotation_6d = _quat_xyzw_to_rotation_matrix(inter_hand_pose[3:7])[:2, :].reshape(6)
    return np.concatenate((inter_hand_pose[:3], rotation_6d, np.asarray(joint_states, dtype=np.float32)))


def _chunk_relative_actions(
    left_poses: np.ndarray, right_poses: np.ndarray, joint_actions: np.ndarray, action_horizon: int
) -> np.ndarray:
    """Turn absolute hand poses for t..t+H into H waypoints relative to the pose at t.

    Row k holds inv(T_t) @ T_{t+1+k} as [xyz, row-major rotation-6D] plus the gripper angle at t+1+k
    (``action.joint_states[t+k]``). Past the end of an episode LeRobot repeats the last frame, so the
    padded rows hold the final pose.
    """
    arm_actions = []
    for arm, arm_poses in enumerate((left_poses, right_poses)):
        poses = np.asarray(arm_poses, dtype=np.float32)
        if poses.shape[0] < action_horizon + 1:
            raise ValueError(
                f"Expected {action_horizon + 1} hand poses (frame t plus the horizon), got {poses.shape[0]}; "
                "the data config must load action_sequence_extra_steps=1."
            )
        current_position = poses[0, :3]
        current_rotation = _quat_xyzw_to_rotation_matrix(poses[0, 3:7])
        target_position = poses[1 : action_horizon + 1, :3]
        target_rotation = _quat_xyzw_to_rotation_matrix(poses[1 : action_horizon + 1, 3:7])

        relative_position = (target_position - current_position) @ current_rotation
        relative_rotation = np.einsum("ji,tjk->tik", current_rotation, target_rotation)
        rotation_6d = relative_rotation[:, :2, :].reshape(-1, 6)
        gripper = np.asarray(joint_actions, dtype=np.float32)[:action_horizon, arm : arm + 1]
        arm_actions.append(np.concatenate((relative_position, rotation_6d, gripper), axis=-1))
    return np.concatenate(arm_actions, axis=-1)


def chunk_to_step_deltas(actions: np.ndarray) -> np.ndarray:
    """Convert (H, 20) chunk-relative waypoints into the (H, 16) UMI Arena publish vector.

    With P_k the waypoint relative to the current pose (P_0 = identity), row k is the body-frame
    delta inv(P_{k-1}) @ P_k, i.e. the same convention as the dataset's ``*.relative`` columns.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] < YUBI_ACTION_DIM:
        raise ValueError(f"Expected actions with shape (H, >={YUBI_ACTION_DIM}), got {actions.shape}")
    horizon = actions.shape[0]
    deltas = np.empty((horizon, YUBI_PUBLISH_DIM), dtype=np.float32)
    for arm in range(2):
        offset = arm * YUBI_ARM_ACTION_DIM
        position = actions[:, offset : offset + 3]
        rotation = _rotation_6d_to_matrix(actions[:, offset + 3 : offset + 9])
        previous_position = np.concatenate((np.zeros((1, 3), dtype=np.float32), position[:-1]))
        previous_rotation = np.concatenate((np.eye(3, dtype=np.float32)[None], rotation[:-1]))

        delta_position = np.einsum("tji,tj->ti", previous_rotation, position - previous_position)
        delta_rotation = np.einsum("tji,tjk->tik", previous_rotation, rotation)
        deltas[:, arm * 7 : arm * 7 + 3] = delta_position
        deltas[:, arm * 7 + 3 : arm * 7 + 7] = _rotation_matrix_to_quat_xyzw(delta_rotation)
        deltas[:, 14 + arm] = np.clip(actions[:, offset + 9], *GRIPPER_LIMITS)
    return deltas


@dataclasses.dataclass(frozen=True)
class YubiUMIInputs(transforms.DataTransformFn):
    """Map a YUBI observation (dataset frame or harness observation) to pi0.5 inputs.

    Only the two wrist cameras exist at evaluation, so the base slot is zero-filled and masked out.
    Actions are built only when the training-only absolute hand poses are present.
    """

    action_horizon: int = 32

    def __call__(self, data: dict) -> dict:
        left_wrist = _parse_image(data[LEFT_IMAGE])
        right_wrist = _parse_image(data[RIGHT_IMAGE])
        inputs = {
            "state": _inter_hand_state(data[INTER_HAND_POSE], data[JOINT_STATES]),
            "image": {
                "base_0_rgb": np.zeros_like(left_wrist),
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if LEFT_HAND_POSE in data:
            inputs["actions"] = _chunk_relative_actions(
                data[LEFT_HAND_POSE], data[RIGHT_HAND_POSE], data[JOINT_ACTIONS], self.action_horizon
            )
        if "prompt" in data:
            inputs["prompt"] = normalize_prompt(data["prompt"])
        return inputs


@dataclasses.dataclass(frozen=True)
class YubiUMIOutputs(transforms.DataTransformFn):
    """Convert the model's chunk-relative actions into the (H, 16) UMI Arena publish vector."""

    def __call__(self, data: dict) -> dict:
        return {"actions": chunk_to_step_deltas(np.asarray(data["actions"])[..., :YUBI_ACTION_DIM])}
