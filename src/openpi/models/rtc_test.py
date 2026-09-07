import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.models import rtc
from openpi.shared import nnx_utils

ACTION_HORIZON = 12
ACTION_DIM = 8


def _make_model(*, pi05: bool, rtc_training_max_delay: int = 0, seed: int = 0):
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        pi05=pi05,
        rtc_training_max_delay=rtc_training_max_delay,
    )
    model = config.create(jax.random.key(seed))
    # The adaRMS modulation is zero-initialised, gating every block to the identity.
    graphdef, state = nnx.split(model)
    keys = iter(jax.random.split(jax.random.key(seed + 100), len(jax.tree.leaves(state))))
    state = jax.tree.map(lambda x: x + 0.15 * jax.random.normal(next(keys), x.shape, x.dtype), state)
    return config, nnx.merge(graphdef, state)


def test_prefix_weights_linear_matches_reference_ramp():
    # linspace(1, 0, n + 2)[1:-1]: the transition excludes both endpoints.
    weights = rtc.get_prefix_weights(3, 7, 12, rtc.PrefixAttentionSchedule.LINEAR)
    np.testing.assert_allclose(weights[:3], 1.0)
    np.testing.assert_allclose(weights[3:7], [0.8, 0.6, 0.4, 0.2], atol=1e-6)
    np.testing.assert_allclose(weights[7:], 0.0)


def test_prefix_weights_hard_schedules():
    ones = rtc.get_prefix_weights(3, 7, 12, rtc.PrefixAttentionSchedule.ONES)
    np.testing.assert_allclose(ones, [1.0] * 7 + [0.0] * 5)
    zeros = rtc.get_prefix_weights(3, 7, 12, rtc.PrefixAttentionSchedule.ZEROS)
    np.testing.assert_allclose(zeros, [1.0] * 3 + [0.0] * 9)


def test_prefix_weights_exp_decays_faster_but_pins_the_endpoints():
    linear = rtc.get_prefix_weights(3, 7, 12, rtc.PrefixAttentionSchedule.LINEAR)
    exp = rtc.get_prefix_weights(3, 7, 12, rtc.PrefixAttentionSchedule.EXP)
    np.testing.assert_allclose(exp[:3], 1.0)
    np.testing.assert_allclose(exp[7:], 0.0)
    assert np.all(exp[3:7] < linear[3:7])


@pytest.mark.parametrize(("start", "end"), [(0, 0), (5, 5), (12, 12), (9, 4), (20, 30)])
def test_prefix_weights_degenerate_ranges_stay_in_bounds(start, end):
    weights = rtc.get_prefix_weights(start, end, 12, rtc.PrefixAttentionSchedule.LINEAR)
    assert weights.shape == (12,)
    assert np.all((weights >= 0.0) & (weights <= 1.0))
    assert np.all(np.diff(weights) <= 1e-6)  # non-increasing


def test_guidance_weight_is_symmetric_and_clipped():
    # Invariant under t <-> 1-t, bottoms out at 2, diverges at both ends.
    assert float(rtc.guidance_weight(0.5, 10.0)) == pytest.approx(2.0)
    assert float(rtc.guidance_weight(0.3, 100.0)) == pytest.approx(float(rtc.guidance_weight(0.7, 100.0)))
    assert float(rtc.guidance_weight(1.0, 10.0)) == pytest.approx(10.0)
    assert float(rtc.guidance_weight(0.001, 10.0)) == pytest.approx(10.0)


@pytest.mark.parametrize("pi05", [False, True])
def test_sampling_without_rtc_is_unchanged(pi05):
    config, model = _make_model(pi05=pi05)
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("rtc_config",))
    obs = config.fake_obs(2)
    noise = jax.random.normal(jax.random.key(1), (2, ACTION_HORIZON, ACTION_DIM))
    prev = jax.random.normal(jax.random.key(2), (2, ACTION_HORIZON, ACTION_DIM))

    baseline = sample(jax.random.key(0), obs, num_steps=4, noise=noise)
    # A previous chunk without a config must not engage RTC.
    assert jnp.array_equal(baseline, sample(jax.random.key(0), obs, num_steps=4, noise=noise, prev_chunk=prev))


@pytest.mark.parametrize("pi05", [False, True])
def test_guided_pulls_the_chunk_towards_the_previous_one(pi05):
    config, model = _make_model(pi05=pi05)
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("rtc_config",))
    obs = config.fake_obs(2)
    noise = jax.random.normal(jax.random.key(1), (2, ACTION_HORIZON, ACTION_DIM))
    prev = jax.random.normal(jax.random.key(2), (2, ACTION_HORIZON, ACTION_DIM))

    def distance(max_guidance_weight):
        # ONES over the full horizon isolates guidance from the weight schedule.
        config_rtc = rtc.RTCConfig(
            mode=rtc.RTCMode.GUIDED,
            prefix_attention_schedule=rtc.PrefixAttentionSchedule.ONES,
            max_guidance_weight=max_guidance_weight,
        )
        actions = sample(
            jax.random.key(0),
            obs,
            num_steps=4,
            noise=noise,
            prev_chunk=prev,
            inference_delay=0,
            execution_horizon=ACTION_HORIZON,
            rtc_config=config_rtc,
        )
        return float(jnp.abs(actions - prev).mean())

    unguided, guided = distance(0.0), distance(10.0)
    assert guided < 0.5 * unguided, f"guidance did not pull the chunk closer: {unguided=} {guided=}"


@pytest.mark.parametrize("pi05", [False, True])
@pytest.mark.parametrize("trained_prefix_noise", [False, True])
def test_trained_mode_pins_the_prefix_exactly(pi05, trained_prefix_noise):
    config, model = _make_model(pi05=pi05)
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("rtc_config",))
    obs = config.fake_obs(2)
    noise = jax.random.normal(jax.random.key(1), (2, ACTION_HORIZON, ACTION_DIM))
    prev = jax.random.normal(jax.random.key(2), (2, ACTION_HORIZON, ACTION_DIM))
    delay = 4

    actions = sample(
        jax.random.key(0),
        obs,
        num_steps=4,
        noise=noise,
        prev_chunk=prev,
        inference_delay=delay,
        execution_horizon=8,
        rtc_config=rtc.RTCConfig(mode=rtc.RTCMode.TRAINED, trained_prefix_noise=trained_prefix_noise),
    )
    np.testing.assert_array_equal(actions[:, :delay], prev[:, :delay])
    assert not jnp.array_equal(actions[:, delay:], prev[:, delay:])


def test_changing_delay_and_horizon_does_not_recompile():
    config, model = _make_model(pi05=True)
    obs = config.fake_obs(1)
    noise = jax.random.normal(jax.random.key(1), (1, ACTION_HORIZON, ACTION_DIM))
    prev = jax.random.normal(jax.random.key(2), (1, ACTION_HORIZON, ACTION_DIM))

    # As nnx_utils.module_jit does, but keeping a handle to read the compilation cache.
    graphdef, state = nnx.split(model)

    def fun(state, *args, **kwargs):
        return nnx.merge(graphdef, state).sample_actions(*args, **kwargs)

    jitted = jax.jit(fun, static_argnames=("rtc_config",))
    for delay, horizon in [(1, 5), (3, 9), (6, 11), (2, 7)]:
        jitted(
            state,
            jax.random.key(0),
            obs,
            num_steps=2,
            noise=noise,
            prev_chunk=prev,
            inference_delay=jnp.asarray(delay, dtype=jnp.int32),
            execution_horizon=jnp.asarray(horizon, dtype=jnp.int32),
            rtc_config=rtc.RTCConfig(),
        )
    # A delay that changes every step must not retrigger compilation.
    assert jitted._cache_size() == 1  # noqa: SLF001


@pytest.mark.parametrize("pi05", [False, True])
def test_training_prefix_is_excluded_from_the_loss(pi05):
    config, model = _make_model(pi05=pi05, rtc_training_max_delay=5, seed=3)
    obs, actions = config.fake_obs(16), config.fake_act(16)
    loss = nnx_utils.module_jit(model.compute_loss)(jax.random.key(7), obs, actions)

    assert loss.shape == (16, ACTION_HORIZON)
    assert bool(jnp.all(jnp.isfinite(loss)))
    # Delays are uniform in [0, 5], so some (but not all) entries are zeroed.
    zeros_per_sample = jnp.sum(loss == 0, axis=-1)
    assert bool(jnp.all(zeros_per_sample <= 5))
    assert int(jnp.sum(zeros_per_sample)) > 0

    _, plain_model = _make_model(pi05=pi05, rtc_training_max_delay=0, seed=3)
    plain_loss = nnx_utils.module_jit(plain_model.compute_loss)(jax.random.key(7), obs, actions)
    assert int(jnp.sum(plain_loss == 0)) == 0


@pytest.mark.parametrize("pi05", [False, True])
def test_training_gradients_are_finite_and_nonzero(pi05):
    config, model = _make_model(pi05=pi05, rtc_training_max_delay=5, seed=3)
    obs, actions = config.fake_obs(4), config.fake_act(4)

    def loss_fn(m):
        return jnp.mean(m.compute_loss(jax.random.key(7), obs, actions, train=True))

    grads = jax.tree.leaves(nnx.grad(loss_fn)(model))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)
    assert any(float(jnp.abs(g).max()) > 0 for g in grads)


def test_rtc_training_max_delay_must_leave_a_supervised_step():
    with pytest.raises(ValueError, match="rtc_training_max_delay"):
        pi0_config.Pi0Config(action_horizon=10, rtc_training_max_delay=10)
