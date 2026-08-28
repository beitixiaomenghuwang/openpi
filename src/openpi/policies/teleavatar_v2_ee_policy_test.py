import numpy as np

from openpi.models import model as _model
from openpi.policies import teleavatar_v2_ee_policy


def _make_record() -> dict:
    half_sqrt = np.float32(np.sqrt(0.5))
    current_pose = np.array([0.4, 0.1, -0.2, 0.0, 0.0, half_sqrt, half_sqrt], dtype=np.float32)
    state = np.zeros(72, dtype=np.float32)
    state[48:55] = current_pose
    state[55:62] = current_pose
    state[7] = 1.0
    state[15] = 1.0

    action = np.zeros((30, 72), dtype=np.float32)
    action[:, 48:55] = current_pose
    action[:, 55:62] = current_pose
    action[:, 48] += np.linspace(0.0, -0.1, len(action), dtype=np.float32)
    action[:, 55] += np.linspace(0.0, 0.1, len(action), dtype=np.float32)
    angle = np.linspace(np.pi / 2.0, np.pi, len(action), dtype=np.float32)
    action[:, 53] = np.sin(angle / 2.0)
    action[:, 54] = np.cos(angle / 2.0)
    action[:, 60] = np.sin(angle / 2.0)
    action[:, 61] = np.cos(angle / 2.0)
    action[:, 39] = np.linspace(2.0, -1.6, len(action), dtype=np.float32)
    action[:, 47] = np.linspace(2.0, -1.6, len(action), dtype=np.float32)
    return {
        "observation/state": state,
        "observation/images/left_color": np.zeros((8, 32, 3), dtype=np.uint8),
        "observation/images/right_color": np.zeros((8, 32, 3), dtype=np.uint8),
        "observation/images/head_camera": np.zeros((8, 16, 3), dtype=np.uint8),
        "action": action,
        "prompt": "test task",
    }


def test_gripper_position_uses_trigger_direction() -> None:
    position = np.array([-0.1, 0.0, 0.25, 1.0, 1.2], dtype=np.float32)
    trigger = teleavatar_v2_ee_policy._gripper_position_to_trigger(position)  # noqa: SLF001
    np.testing.assert_allclose(trigger, [1.0, 1.0, 0.75, 0.0, 0.0])


def test_input_output_pose_and_gripper_semantics() -> None:
    record = _make_record()
    transformed = teleavatar_v2_ee_policy.TeleavatarEEInputs(
        model_type=_model.ModelType.PI0,
    )(record)

    assert transformed["state"].shape == (20,)
    assert transformed["actions"].shape == (30, 20)
    assert transformed["state"][9] == 0.0
    assert transformed["state"][19] == 0.0
    np.testing.assert_allclose(transformed["actions"][[0, -1], 9], [0.0, 1.0])
    np.testing.assert_allclose(transformed["actions"][[0, -1], 19], [0.0, 1.0])

    decoded = teleavatar_v2_ee_policy.TeleavatarEEOutputs()(
        {"state": transformed["state"], "actions": transformed["actions"]}
    )["actions"]
    expected_left_pose = teleavatar_v2_ee_policy._matrix_to_pose10(  # noqa: SLF001
        teleavatar_v2_ee_policy._pose7_to_matrix(record["action"][:, 48:55])  # noqa: SLF001
    )
    expected_right_pose = teleavatar_v2_ee_policy._matrix_to_pose10(  # noqa: SLF001
        teleavatar_v2_ee_policy._pose7_to_matrix(record["action"][:, 55:62])  # noqa: SLF001
    )
    np.testing.assert_allclose(decoded[:, :9], expected_left_pose, atol=1e-6)
    np.testing.assert_allclose(decoded[:, 10:19], expected_right_pose, atol=1e-6)
    np.testing.assert_allclose(decoded[:, 9], transformed["actions"][:, 9])
    np.testing.assert_allclose(decoded[:, 19], transformed["actions"][:, 19])


def test_v2_camera_crop_matches_existing_policy() -> None:
    transformed = teleavatar_v2_ee_policy.TeleavatarEEInputs(
        model_type=_model.ModelType.PI0,
    )(_make_record())
    assert transformed["image"]["base_0_rgb"].shape == (8, 8, 3)
    assert transformed["image"]["left_wrist_0_rgb"].shape == (8, 16, 3)
    assert transformed["image"]["right_wrist_0_rgb"].shape == (8, 16, 3)
