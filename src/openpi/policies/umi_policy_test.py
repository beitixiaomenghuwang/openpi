import numpy as np

from openpi.policies import umi_policy


def test_umi_inputs_use_images_and_relative_row_major_rotation_6d():
    state = np.array(
        [
            1.0,
            2.0,
            3.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            4.0,
            5.0,
            6.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )
    # Left target has a +90 degree rotation around z; right target is unchanged.
    action = np.array(
        [[2.0, 4.0, 6.0, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5), 1.0, *state[8:]]], dtype=np.float32
    )
    data = {
        "observation/state": state.copy(),
        "observation/images/head_camera": np.zeros((3, 4, 5), dtype=np.float32),
        "observation/images/left_color": np.zeros((4, 5, 3), dtype=np.uint8),
        "observation/images/right_color": np.zeros((4, 5, 3), dtype=np.uint8),
        "action": action,
        "prompt": "move the arms",
    }

    result = umi_policy.UMIInputs()(data)

    np.testing.assert_array_equal(result["state"], np.zeros(1, dtype=np.float32))
    assert result["image"]["base_0_rgb"].shape == (4, 5, 3)
    np.testing.assert_allclose(
        result["actions"][0, :10],
        [1.0, 2.0, 3.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["actions"][0, 10:],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        atol=1e-6,
    )
    np.testing.assert_array_equal(data["observation/state"], state)


def test_umi_inputs_pad_and_mask_the_base_slot_without_a_head_camera():
    data = {
        "observation/state": np.zeros(16, dtype=np.float32),
        "observation/images/left_color": np.full((4, 5, 3), 7, dtype=np.uint8),
        "observation/images/right_color": np.full((4, 5, 3), 9, dtype=np.uint8),
    }

    result = umi_policy.UMIInputs(use_head_camera=False)(data)

    np.testing.assert_array_equal(result["image"]["base_0_rgb"], np.zeros((4, 5, 3), dtype=np.uint8))
    assert not result["image_mask"]["base_0_rgb"]
    assert result["image_mask"]["left_wrist_0_rgb"]
    assert result["image_mask"]["right_wrist_0_rgb"]


def test_umi_outputs_return_the_20_dimensional_action():
    actions = np.arange(64, dtype=np.float32).reshape(2, 32)

    result = umi_policy.UMIOutputs()({"actions": actions})

    np.testing.assert_array_equal(result["actions"], actions[:, :20])


def test_relative_action_reconstructs_absolute_pose():
    state = np.array(
        [0.2, -0.1, 0.4, 0.0, 0.0, 0.0, 1.0, 1.0, -0.3, 0.5, 0.1, 0.0, 0.0, 0.0, 1.0, 1.0],
        dtype=np.float32,
    )
    absolute = np.array(
        [0.3, 0.1, 0.5, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5), 0.0,
         -0.2, 0.4, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0],
        dtype=np.float32,
    )
    relative = umi_policy._relative_ee_actions(state, absolute[None])[0]  # noqa: SLF001
    reconstructed = umi_policy._relative_to_absolute_ee_action(state, relative)  # noqa: SLF001

    np.testing.assert_allclose(reconstructed[:7], absolute[:7], atol=1e-5)
    np.testing.assert_allclose(reconstructed[8:15], absolute[8:15], atol=1e-5)
    np.testing.assert_array_equal(reconstructed[[7, 15]], [1.0, 0.0])

    chunk = umi_policy.relative_actions_to_absolute(state, np.stack((relative, relative)))
    np.testing.assert_allclose(chunk[0], reconstructed, atol=1e-5)
    np.testing.assert_allclose(chunk[1], reconstructed, atol=1e-5)
