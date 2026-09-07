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
from openpi.policies import teleavatar_v2_policy
from openpi.shared import normalize as _normalize
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


# --- RTC with delta actions ---------------------------------------------------------


def _make_delta_rtc_policy(*, use_delta_joint_actions: bool) -> _policy.Policy:
    """A real teleavatar transform stack (delta encoding, non-trivial norm stats) on a dummy
    model, so the re-anchoring can be checked end to end in physical units."""
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_dim=32,
        action_horizon=_RTC_ACTION_HORIZON,
    )
    model = config.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    keys = iter(jax.random.split(jax.random.key(100), len(jax.tree.leaves(state))))
    state = jax.tree.map(lambda x: x + 0.15 * jax.random.normal(next(keys), x.shape, x.dtype), state)

    # Deliberately non-unit, so a re-anchoring that forgets to scale into normalized space
    # cannot pass.
    rng = np.random.default_rng(0)
    norm_stats = {
        "state": _normalize.NormStats(mean=rng.normal(size=14), std=rng.uniform(0.5, 2.0, size=14)),
        "actions": _normalize.NormStats(mean=rng.normal(size=16), std=rng.uniform(0.5, 2.0, size=16)),
    }
    model_transforms = _config.ModelTransformFactory()(config)
    return _policy.Policy(
        nnx.merge(graphdef, state),
        transforms=[
            teleavatar_v2_policy.TeleavatarInputs(
                model_type=config.model_type, use_delta_joint_actions=use_delta_joint_actions
            ),
            _transforms.Normalize(norm_stats),
            *model_transforms.inputs,
        ],
        output_transforms=[
            *model_transforms.outputs,
            _transforms.Unnormalize(norm_stats),
            teleavatar_v2_policy.TeleavatarOutputs(use_delta_joint_actions=use_delta_joint_actions),
        ],
        rtc_config=_rtc.RTCConfig(mode=_rtc.RTCMode.TRAINED),
    )


def _teleavatar_example(arm_offset: float) -> dict:
    state = np.zeros(62, np.float32)
    state[0:16] = arm_offset  # joint positions; the rest (velocities, efforts, ee pose) are unused
    return {
        "observation/state": state,
        "observation/images/left_color": np.zeros((400, 640, 3), np.uint8),
        "observation/images/right_color": np.zeros((400, 640, 3), np.uint8),
        "observation/images/head_camera": np.zeros((960, 960, 3), np.uint8),
        "prompt": "do the thing",
    }


@pytest.mark.parametrize("use_delta_joint_actions", [False, True])
def test_rtc_prefix_matches_in_physical_space_across_a_state_change(use_delta_joint_actions):
    # TRAINED mode reproduces the pinned prefix exactly, so with correct re-anchoring the
    # two chunks must agree on the overlap in *absolute joint* units even though the robot
    # moved in between. Without it they would differ by exactly that motion.
    policy = _make_delta_rtc_policy(use_delta_joint_actions=use_delta_joint_actions)
    prefix_start, delay = 3, 4

    first = policy.infer(_teleavatar_example(0.0))
    second = policy.infer(
        {
            **_teleavatar_example(0.3),  # the arm has moved a long way since the first chunk
            _policy.RTC_PREV_CHUNK_ID: 1,
            _policy.RTC_PREFIX_START: prefix_start,
            _policy.RTC_INFERENCE_DELAY: delay,
            _policy.RTC_EXECUTION_HORIZON: 8,
        }
    )
    np.testing.assert_allclose(
        second["actions"][:delay],
        first["actions"][prefix_start : prefix_start + delay],
        rtol=1e-4,
        atol=1e-4,
    )


def test_delta_anchor_is_only_produced_for_delta_actions():
    absolute = _make_delta_rtc_policy(use_delta_joint_actions=False)
    delta = _make_delta_rtc_policy(use_delta_joint_actions=True)
    example = _teleavatar_example(0.3)

    assert absolute._rtc_anchor(example) is None  # noqa: SLF001
    anchor = delta._rtc_anchor(example)  # noqa: SLF001
    assert anchor is not None
    # Grippers stay absolute, so their anchor is 0 before normalization -- after it, it is
    # whatever the norm stats map 0 to, which is what the difference of two anchors cancels.
    assert anchor.shape == (16,)


def test_delta_actions_transform_reports_its_anchor():
    mask = _transforms.make_bool_mask(3, -2)
    transform = _transforms.DeltaActions(mask)
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(transform.delta_action_anchor({"state": state}), [1.0, 2.0, 3.0, 0.0, 0.0])
    assert _transforms.DeltaActions(None).delta_action_anchor({"state": state}) is None
