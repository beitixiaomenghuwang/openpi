import dataclasses

import flax.nnx as nnx
import jax
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import rtc as _rtc
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)


# --- Real-Time Chunking ---------------------------------------------------------------

_RTC_ACTION_HORIZON = 12
_RTC_ACTION_DIM = 8


def _make_rtc_policy(mode: _rtc.RTCMode, extra_transforms=()) -> _policy.Policy:
    """A tiny end-to-end policy: dummy weights, no transforms, RTC enabled."""
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_dim=_RTC_ACTION_DIM,
        action_horizon=_RTC_ACTION_HORIZON,
        pi05=True,
    )
    model = config.create(jax.random.key(0))
    # The adaRMS modulation is zero-initialised, gating every block to the identity.
    graphdef, state = nnx.split(model)
    keys = iter(jax.random.split(jax.random.key(100), len(jax.tree.leaves(state))))
    state = jax.tree.map(lambda x: x + 0.15 * jax.random.normal(next(keys), x.shape, x.dtype), state)
    return _policy.Policy(
        nnx.merge(graphdef, state),
        transforms=list(extra_transforms),
        rtc_config=_rtc.RTCConfig(mode=mode),
    )


def _rtc_example() -> dict:
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    return {
        "image": {k: np.zeros((*_model.IMAGE_RESOLUTION, 3), np.float32) for k in keys},
        "image_mask": dict.fromkeys(keys, np.True_),
        "state": np.zeros(_RTC_ACTION_DIM, np.float32),
        "tokenized_prompt": np.zeros(200, np.int32),
        "tokenized_prompt_mask": np.ones(200, bool),
    }


def test_rtc_policy_aligns_the_previous_chunk_to_the_new_one():
    # TRAINED reproduces the pinned prefix exactly, making the server-side alignment
    # directly observable.
    policy = _make_rtc_policy(_rtc.RTCMode.TRAINED)
    example = _rtc_example()

    first = policy.infer(example)
    assert first[_policy.RTC_CHUNK_ID] == 1
    assert first[_policy.RTC_ACTION_HORIZON] == _RTC_ACTION_HORIZON

    prefix_start, delay = 5, 4
    second = policy.infer(
        {
            **example,
            _policy.RTC_PREV_CHUNK_ID: 1,
            _policy.RTC_PREFIX_START: prefix_start,
            _policy.RTC_INFERENCE_DELAY: delay,
            _policy.RTC_EXECUTION_HORIZON: 8,
        }
    )
    assert second[_policy.RTC_CHUNK_ID] == 2
    np.testing.assert_allclose(
        second["actions"][:delay], first["actions"][prefix_start : prefix_start + delay], rtol=1e-5, atol=1e-5
    )
    # Past the prefix the chunk is free again. Compare only where the two still overlap.
    overlap = len(first["actions"]) - prefix_start
    assert not np.allclose(second["actions"][delay:overlap], first["actions"][prefix_start + delay :])


def test_rtc_policy_first_chunk_and_reset_are_unconstrained():
    policy = _make_rtc_policy(_rtc.RTCMode.GUIDED)
    example = _rtc_example()

    # prev_chunk_id 0 means "nothing executing": must not raise despite the empty history.
    assert policy.infer({**example, _policy.RTC_PREV_CHUNK_ID: 0})[_policy.RTC_CHUNK_ID] == 1
    policy.reset()
    assert policy.infer({**example, _policy.RTC_PREV_CHUNK_ID: 0})[_policy.RTC_CHUNK_ID] == 1


def test_rtc_policy_rejects_a_stale_chunk_id():
    policy = _make_rtc_policy(_rtc.RTCMode.GUIDED)
    example = _rtc_example()
    policy.infer(example)
    with pytest.raises(RuntimeError, match="chunk id mismatch"):
        policy.infer({**example, _policy.RTC_PREV_CHUNK_ID: 7, _policy.RTC_PREFIX_START: 0})


def test_rtc_fields_never_reach_the_data_transforms():
    seen: list[set] = []

    @dataclasses.dataclass(frozen=True)
    class _Spy(_transforms.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            seen.append(set(data))
            return data

    policy = _make_rtc_policy(_rtc.RTCMode.GUIDED, extra_transforms=[_Spy()])
    example = _rtc_example()
    policy.infer(example)
    policy.infer({**example, _policy.RTC_PREV_CHUNK_ID: 1, _policy.RTC_PREFIX_START: 3})

    assert seen, "the spy transform never ran"
    assert all(not any(k.startswith("rtc/") for k in keys) for keys in seen)
