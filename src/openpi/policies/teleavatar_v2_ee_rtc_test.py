import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import teleavatar_v2_ee_policy as _ee


def _pose_matrix(xyz: tuple[float, float, float], quaternion: tuple[float, float, float, float]) -> np.ndarray:
    pose = np.asarray((*xyz, *quaternion), dtype=np.float32)
    return _ee._pose7_to_matrix(pose)


def _make_action(relative: np.ndarray, trigger: float, *, padded_dim: int = 32) -> np.ndarray:
    action = np.zeros((3, padded_dim), dtype=np.float32)
    for start in (0, _ee.EE_ARM_ACTION_DIM):
        action[:, start : start + 9] = _ee._matrix_to_pose10(relative)
        action[:, start + 9] = trigger
    action[:, _ee.EE_ACTION_DIM :] = 7.0
    return action


def test_ee_transform_implements_rtc_reanchor_protocol():
    transform = _ee.TeleavatarEEInputs(model_type=_model.ModelType.PI05)
    assert isinstance(transform, transforms.RTCActionReanchor)


def test_ee_reanchor_preserves_absolute_targets_for_both_arms():
    transform = _ee.TeleavatarEEInputs(model_type=_model.ModelType.PI05)
    previous = np.stack(
        [
            _pose_matrix((0.30, 0.00, -0.30), (0.0, 0.0, 0.0, 1.0)),
            _pose_matrix((0.30, 0.50, -0.30), (0.0, 0.0, 0.0, 1.0)),
        ]
    )
    current = np.stack(
        [
            _pose_matrix((0.37, -0.04, -0.27), (0.0, 0.0, 0.38268343, 0.9238795)),
            _pose_matrix((0.24, 0.56, -0.31), (0.0, 0.0, -0.258819, 0.9659258)),
        ]
    )
    targets = np.stack(
        [
            _pose_matrix((0.48, 0.08, -0.18), (0.0, 0.0, 0.130526, 0.991445)),
            _pose_matrix((0.17, 0.62, -0.22), (0.0, 0.0, -0.130526, 0.991445)),
        ]
    )

    old_relative = np.stack([np.linalg.inv(previous[i]) @ targets[i] for i in range(2)])
    actions = _make_action(old_relative[0], trigger=0.4)
    actions[:, _ee.EE_ARM_ACTION_DIM : _ee.EE_ARM_ACTION_DIM + 9] = _ee._matrix_to_pose10(old_relative[1])

    reanchored = transform.rtc_reanchor(actions, previous, current)
    for start, arm in ((0, 0), (_ee.EE_ARM_ACTION_DIM, 1)):
        new_relative = _ee._pose10_to_matrix(reanchored[:, start : start + 9])
        np.testing.assert_allclose(
            current[arm] @ new_relative,
            np.broadcast_to(targets[arm], new_relative.shape),
            rtol=1e-5,
            atol=1e-5,
        )
    np.testing.assert_array_equal(reanchored[:, _ee.EE_ACTION_DIM :], 7.0)
    np.testing.assert_array_equal(reanchored[:, 9], 0.4)
    np.testing.assert_array_equal(reanchored[:, 19], 0.4)
