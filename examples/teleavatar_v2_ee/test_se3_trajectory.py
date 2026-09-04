import numpy as np

from examples.teleavatar_v2_ee.se3_trajectory import SE3Trajectory
from examples.teleavatar_v2_ee.se3_trajectory import _quaternion_conjugate
from examples.teleavatar_v2_ee.se3_trajectory import _quaternion_multiply
from examples.teleavatar_v2_ee.se3_trajectory import _quaternion_to_rotvec


def _quaternion_z(angle: float) -> np.ndarray:
    return np.array((0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)), dtype=np.float64)


def _action(position: float = 0.0, angle: float = 0.0) -> np.ndarray:
    quaternion = _quaternion_z(angle)
    return np.array(
        (
            position,
            0.0,
            0.0,
            *quaternion,
            0.0,
            position,
            0.0,
            0.0,
            *quaternion,
            0.0,
        ),
        dtype=np.float64,
    )


def _trajectory() -> SE3Trajectory:
    return SE3Trajectory(
        minimum_duration=1.0 / 45.0,
        max_translation_speed=0.25,
        max_rotation_speed=1.2,
        max_translation_acceleration=1.0,
        max_rotation_acceleration=4.0,
        max_translation_jerk=10.0,
        max_rotation_jerk=40.0,
    )


def test_shortest_path_is_sign_invariant() -> None:
    start = _action()
    target = _action(0.02, np.deg2rad(170.0))
    signed_target = target.copy()
    signed_target[3:7] *= -1.0
    signed_target[11:15] *= -1.0

    positive_trajectory = _trajectory()
    positive_trajectory.reset(start)
    positive_trajectory.set_target(target)
    positive_output = positive_trajectory.step(0.005)

    signed_trajectory = _trajectory()
    signed_trajectory.reset(start)
    signed_trajectory.set_target(signed_target)
    output = signed_trajectory.step(0.005)

    assert np.isclose(np.linalg.norm(output[3:7]), 1.0)
    assert np.isclose(np.linalg.norm(output[11:15]), 1.0)
    np.testing.assert_allclose(output, positive_output, atol=1e-12)


def test_retargeted_stream_respects_se3_limits() -> None:
    trajectory = _trajectory()
    trajectory.reset(_action())
    outputs = []
    for tick in range(1000):
        trajectory.set_target(_action(0.15, 1.0) if tick < 50 else _action(-0.1, -1.0))
        outputs.append(trajectory.step(0.005))
    outputs = np.asarray(outputs)

    dt = 0.005
    translation_velocity = np.diff(outputs[:, :3], axis=0) / dt
    translation_acceleration = np.diff(translation_velocity, axis=0) / dt
    translation_jerk = np.diff(translation_acceleration, axis=0) / dt
    assert np.linalg.norm(translation_velocity, axis=1).max() <= 0.25 + 1e-6
    assert np.linalg.norm(translation_acceleration, axis=1).max() <= 1.0 + 1e-6
    assert np.linalg.norm(translation_jerk, axis=1).max() <= 10.0 + 1e-6

    angular_velocity = []
    for first, second in zip(outputs[:-1, 3:7], outputs[1:, 3:7]):
        angular_velocity.append(_quaternion_to_rotvec(_quaternion_multiply(second, _quaternion_conjugate(first))) / dt)
    angular_velocity = np.asarray(angular_velocity)
    angular_acceleration = np.diff(angular_velocity, axis=0) / dt
    angular_jerk = np.diff(angular_acceleration, axis=0) / dt
    assert np.linalg.norm(angular_velocity, axis=1).max() <= 1.2 + 1e-6
    assert np.linalg.norm(angular_acceleration, axis=1).max() <= 4.0 + 1e-6
    assert np.linalg.norm(angular_jerk, axis=1).max() <= 40.0 + 1e-6

    np.testing.assert_allclose(outputs[-1][:3], (-0.1, 0.0, 0.0), atol=1e-5)
    np.testing.assert_allclose(outputs[-1][3:7], _quaternion_z(-1.0), atol=1e-5)
