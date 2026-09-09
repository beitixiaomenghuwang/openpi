import numpy as np

from examples.teleavatar_v2_ee.se3_trajectory import SE3Interpolator


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


def test_shortest_path_is_sign_invariant() -> None:
    start = _action()
    target = _action(0.02, np.deg2rad(170.0))
    signed_target = target.copy()
    signed_target[3:7] *= -1.0
    signed_target[11:15] *= -1.0

    positive_interpolator = SE3Interpolator(duration=1.0 / 45.0)
    positive_interpolator.reset(start)
    positive_interpolator.set_target(target)
    positive_output = positive_interpolator.step(0.005)

    signed_interpolator = SE3Interpolator(duration=1.0 / 45.0)
    signed_interpolator.reset(start)
    signed_interpolator.set_target(signed_target)
    output = signed_interpolator.step(0.005)

    assert np.isclose(np.linalg.norm(output[3:7]), 1.0)
    assert np.isclose(np.linalg.norm(output[11:15]), 1.0)
    np.testing.assert_allclose(output, positive_output, atol=1e-12)


def test_reaches_each_target_within_one_control_period() -> None:
    """The interpolator upsamples the target stream without adding tracking lag."""
    control_period = 1.0 / 45.0
    interpolator = SE3Interpolator(duration=control_period)
    interpolator.reset(_action())
    target = _action(0.02, np.deg2rad(30.0))
    interpolator.set_target(target)

    outputs = [interpolator.step(0.005) for _ in range(9)]  # 45 ms > one 22.2 ms control period
    np.testing.assert_allclose(outputs[-1][:3], target[:3], atol=1e-12)
    np.testing.assert_allclose(outputs[-1][3:7], target[3:7], atol=1e-12)
    # The traversal is exactly linear in elapsed time, so 10 ms into a 22.2 ms
    # period the command sits at 45% of the way, not behind it.
    assert np.isclose(outputs[1][0] / target[0], 0.010 / control_period, rtol=1e-9)
    for output in outputs:
        assert np.isclose(np.linalg.norm(output[3:7]), 1.0)


def test_command_stream_is_not_rate_limited() -> None:
    """A large model jump is passed through, not clipped to a speed bound."""
    control_period = 1.0 / 45.0
    interpolator = SE3Interpolator(duration=control_period)
    interpolator.reset(_action())
    interpolator.set_target(_action(0.10))  # 10 cm in one control period == 4.5 m/s

    outputs = [interpolator.step(0.005) for _ in range(5)]  # 25 ms, about one control period
    speeds = np.linalg.norm(np.diff(np.asarray(outputs)[:, :3], axis=0), axis=1) / 0.005
    # The command moves at exactly the rate the target implies, with no clipping.
    # (The final tick is slower only because the traversal completes inside it.)
    assert np.isclose(speeds.max(), 0.10 / control_period)


def test_retarget_resumes_from_the_published_command() -> None:
    """Replacing the target mid-traversal must not teleport the stream."""
    interpolator = SE3Interpolator(duration=1.0 / 45.0)
    interpolator.reset(_action())
    outputs = []
    for tick in range(200):
        if tick % 4 == 0:  # a new target every 4 ticks, as at 45 Hz targets / 200 Hz publishing
            interpolator.set_target(_action(0.15, 1.0) if tick < 52 else _action(-0.1, -1.0))
        outputs.append(interpolator.step(0.005))
    outputs = np.asarray(outputs)

    # Tick 52 reverses the target by 0.25 m. The command continues from the one
    # just published, covering exactly dt/duration of the new gap rather than
    # jumping to the new target.
    before, after = outputs[51][0], outputs[52][0]
    alpha = 0.005 / (1.0 / 45.0)
    assert np.isclose(after, before + alpha * (-0.1 - before), rtol=1e-9)
    np.testing.assert_allclose(outputs[-1][:3], (-0.1, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(outputs[-1][3:7], _quaternion_z(-1.0), atol=1e-6)
    for output in outputs:
        assert np.isclose(np.linalg.norm(output[3:7]), 1.0)
