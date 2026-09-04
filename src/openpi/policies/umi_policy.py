"""Policy transforms for UMI dual-arm end-effector data."""

import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _quat_xyzw_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized xyzw quaternions to rotation matrices."""
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
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _relative_ee_actions(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Convert absolute UMI EE targets to per-arm relative xyz + row-major 6D rotation."""
    state = np.asarray(state, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    arm_actions = []
    for offset in (0, 8):
        current_position = state[offset : offset + 3]
        current_rotation = _quat_xyzw_to_rotation_matrix(state[offset + 3 : offset + 7])
        target_position = actions[:, offset : offset + 3]
        target_rotation = _quat_xyzw_to_rotation_matrix(actions[:, offset + 3 : offset + 7])

        relative_position = (target_position - current_position) @ current_rotation
        relative_rotation = np.einsum("ji,tjk->tik", current_rotation, target_rotation)
        rotation_6d = relative_rotation[:, :2, :].reshape(-1, 6)
        # UMI stores 1=open/0=closed; the model convention is 0=open/1=closed.
        gripper = 1.0 - actions[:, offset + 7 : offset + 8]
        arm_actions.append(np.concatenate((relative_position, rotation_6d, gripper), axis=-1))
    return np.concatenate(arm_actions, axis=-1)


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
    """Convert rotation matrices to xyzw quaternions."""
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
            s = np.sqrt(np.maximum(1.0 + rotation[selected, 0, 0] - rotation[selected, 1, 1] - rotation[selected, 2, 2], 1e-8)) * 2
            result[selected, 0] = 0.25 * s
            result[selected, 1] = (rotation[selected, 0, 1] + rotation[selected, 1, 0]) / s
            result[selected, 2] = (rotation[selected, 0, 2] + rotation[selected, 2, 0]) / s
            result[selected, 3] = (rotation[selected, 2, 1] - rotation[selected, 1, 2]) / s
        elif index == 1:
            s = np.sqrt(np.maximum(1.0 + rotation[selected, 1, 1] - rotation[selected, 0, 0] - rotation[selected, 2, 2], 1e-8)) * 2
            result[selected, 0] = (rotation[selected, 0, 1] + rotation[selected, 1, 0]) / s
            result[selected, 1] = 0.25 * s
            result[selected, 2] = (rotation[selected, 1, 2] + rotation[selected, 2, 1]) / s
            result[selected, 3] = (rotation[selected, 0, 2] - rotation[selected, 2, 0]) / s
        else:
            s = np.sqrt(np.maximum(1.0 + rotation[selected, 2, 2] - rotation[selected, 0, 0] - rotation[selected, 1, 1], 1e-8)) * 2
            result[selected, 0] = (rotation[selected, 0, 2] + rotation[selected, 2, 0]) / s
            result[selected, 1] = (rotation[selected, 1, 2] + rotation[selected, 2, 1]) / s
            result[selected, 2] = 0.25 * s
            result[selected, 3] = (rotation[selected, 1, 0] - rotation[selected, 0, 1]) / s
    result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-8)
    return result.reshape(*original_shape, 4)


def _relative_to_absolute_ee_action(current_state: np.ndarray, relative_action: np.ndarray) -> np.ndarray:
    """Combine current UMI EE state with one 20-D relative action waypoint."""
    current_state = np.asarray(current_state, dtype=np.float32)
    relative_action = np.asarray(relative_action, dtype=np.float32)
    if current_state.shape != (16,) or relative_action.shape != (20,):
        raise ValueError(f"Expected current state (16,) and relative action (20,), got {current_state.shape=} {relative_action.shape=}")

    absolute_arms = []
    for state_offset, action_offset in ((0, 0), (8, 10)):
        current_position = current_state[state_offset : state_offset + 3]
        current_rotation = _quat_xyzw_to_rotation_matrix(current_state[state_offset + 3 : state_offset + 7])
        relative_position = relative_action[action_offset : action_offset + 3]
        relative_rotation = _rotation_6d_to_matrix(relative_action[action_offset + 3 : action_offset + 9])
        target_position = current_position + current_rotation @ relative_position
        target_rotation = current_rotation @ relative_rotation
        target_quaternion = _rotation_matrix_to_quat_xyzw(target_rotation)
        gripper = relative_action[action_offset + 9 : action_offset + 10]
        absolute_arms.append(np.concatenate((target_position, target_quaternion, gripper)))
    return np.concatenate(absolute_arms)


def relative_actions_to_absolute(current_state: np.ndarray, relative_actions: np.ndarray) -> np.ndarray:
    """Convert an entire action chunk using one current-state snapshot."""
    relative_actions = np.asarray(relative_actions, dtype=np.float32)
    if relative_actions.ndim != 2 or relative_actions.shape[-1] != 20:
        raise ValueError(f"Expected relative actions with shape (T, 20), got {relative_actions.shape}")
    return np.stack(
        [_relative_to_absolute_ee_action(current_state, action) for action in relative_actions], axis=0
    )


@dataclasses.dataclass(frozen=True)
class UMIInputs(transforms.DataTransformFn):
    """Map UMI's three monocular cameras and absolute EE actions to pi0.5 inputs."""

    def __call__(self, data: dict) -> dict:
        inputs = {
            # Pi0.5 ignores continuous state when discrete_state_input=False.
            # Keep this zero placeholder for the shared Observation schema.
            "state": np.zeros(1, dtype=np.float32),
            "image": {
                "base_0_rgb": _parse_image(data["observation/images/head_camera"]),
                "left_wrist_0_rgb": _parse_image(data["observation/images/left_color"]),
                "right_wrist_0_rgb": _parse_image(data["observation/images/right_color"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "action" in data:
            inputs["actions"] = _relative_ee_actions(data["observation/state"], data["action"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class UMIOutputs(transforms.DataTransformFn):
    """Return UMI's 20-D relative end-effector action representation."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :20])}
