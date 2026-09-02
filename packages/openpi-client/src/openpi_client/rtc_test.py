"""Tests for the client side of Real-Time Chunking.

The property that matters is splice alignment: every control step executes exactly one
action, with no gap and no repeat when a new chunk replaces the one in flight. These drive
the broker against a fake latency-simulating policy and reconstruct, for every executed
action, which control step it was predicted for.
"""

import time

import numpy as np
import pytest

from openpi_client import rtc

ACTION_HORIZON = 30


class _FakePolicy:
    """Returns chunks whose action value encodes the index within the chunk."""

    def __init__(self, latency: float):
        self._latency = latency
        self.calls = 0
        self.requests: list[dict] = []

    def infer(self, obs: dict) -> dict:
        self.requests.append({k: v for k, v in obs.items() if k.startswith("rtc/")})
        time.sleep(self._latency)
        self.calls += 1
        return {
            "actions": np.arange(ACTION_HORIZON, dtype=np.float64)[:, None].repeat(4, axis=1),
            rtc.RTC_CHUNK_ID: self.calls,
            rtc.RTC_ACTION_HORIZON: ACTION_HORIZON,
            # Echoed back so the test can reconstruct chunk alignment.
            "req_prefix_start": int(obs[rtc.RTC_PREFIX_START]),
            "req_prev_id": int(obs[rtc.RTC_PREV_CHUNK_ID]),
        }

    def reset(self) -> None:
        pass


def _run(latency: float, control_hz: float, n_steps: int, **config_kwargs):
    policy = _FakePolicy(latency)
    config_kwargs.setdefault("warmup_steps", 1)
    config_kwargs.setdefault("calibration_steps", 3)
    broker = rtc.RTCActionBroker(policy, rtc.RTCBrokerConfig(control_frequency=control_hz, **config_kwargs))

    step_time = 1.0 / control_hz
    offsets: dict[int, int] = {}  # chunk id -> global control step of that chunk's index 0
    executed = []
    deadline = time.perf_counter()
    for step in range(n_steps):
        result = broker.infer({"observation": step})
        index = int(result["actions"][0])
        chunk_id = result[rtc.RTC_CHUNK_ID]
        if chunk_id not in offsets:
            # Warm-start chunks are all aligned with step 0 (same observation, prefix_start 0)
            # and are absent from `offsets` because none of them was executed.
            previous = result["req_prev_id"]
            offsets[chunk_id] = offsets.get(previous, 0) + result["req_prefix_start"]
        executed.append((step, chunk_id, offsets[chunk_id] + index))
        # Pace like Runtime._run_episode: never catch up a deficit, or the blocking warm
        # start on step 0 is followed by a burst of free steps.
        deadline = max(deadline + step_time, time.perf_counter())
        time.sleep(max(0.0, deadline - time.perf_counter()))
    return broker, policy, executed


def _assert_aligned(executed):
    mismatches = [(step, target) for step, _, target in executed if step != target]
    assert not mismatches, f"action/step misalignment at {mismatches[:5]}"
    chunk_ids = [chunk_id for _, chunk_id, _ in executed]
    assert len(set(chunk_ids)) > 1, "test did not exercise a chunk splice"


@pytest.mark.parametrize(("latency", "control_hz"), [(0.12, 20.0), (0.40, 20.0)])
def test_splices_execute_every_step_exactly_once(latency, control_hz):
    _, policy, executed = _run(latency, control_hz, 90)
    _assert_aligned(executed)
    constrained = [r for r in policy.requests if r[rtc.RTC_PREV_CHUNK_ID] != 0]
    assert constrained, "no RTC-constrained request was issued"
    assert all(0 <= r[rtc.RTC_PREFIX_START] < ACTION_HORIZON for r in constrained)
    assert all(r[rtc.RTC_INFERENCE_DELAY] < r[rtc.RTC_EXECUTION_HORIZON] for r in constrained)


def test_calibration_sizes_the_delay_from_measured_latency():
    # 120 ms at 20 Hz is 2.4 steps -> ceil 3, plus the default 1 step of margin.
    # adapt_delay off so this reflects calibration alone.
    broker, _, _ = _run(0.12, 20.0, 40, adapt_delay=False)
    assert broker.inference_delay == 4
    assert broker.execution_horizon == 8  # 2x the delay by default


def test_explicit_delay_and_horizon_skip_calibration():
    broker, policy, executed = _run(
        0.12, 20.0, 60, calibration_steps=0, inference_delay=5, execution_horizon=12, adapt_delay=False
    )
    assert (broker.inference_delay, broker.execution_horizon) == (5, 12)
    # Only the single bootstrap inference runs before the first action is executed.
    assert policy.requests[0][rtc.RTC_PREV_CHUNK_ID] == 0
    _assert_aligned(executed)


def test_warm_start_exercises_the_constrained_path():
    # The server traces a different graph once a previous chunk is attached, so the warm
    # start has to trigger that compile and measure on it. Measuring the unconstrained path
    # instead both under-estimates the delay and leaves the compile for the first real
    # request, which then overruns the queue.
    _, policy, _ = _run(0.05, 20.0, 25)
    warm_start = policy.requests[:4]  # warmup_steps=1 + calibration_steps=3
    assert warm_start[0][rtc.RTC_PREV_CHUNK_ID] == 0, "the first query has nothing to constrain against"
    assert all(r[rtc.RTC_PREV_CHUNK_ID] != 0 for r in warm_start[1:])


def test_nothing_is_executed_until_the_warm_start_finishes():
    policy = _FakePolicy(0.05)
    broker = rtc.RTCActionBroker(
        policy, rtc.RTCBrokerConfig(control_frequency=20.0, warmup_steps=2, calibration_steps=3)
    )
    assert policy.calls == 0
    broker.infer({"observation": 0})  # the very first control step blocks through calibration
    assert policy.calls == 5, "warm start must consume warmup_steps + calibration_steps inferences"


def test_reset_reuses_the_calibration():
    # Runtime calls agent.reset() at every episode boundary. Re-measuring there costs a
    # multiple of the inference time during which the robot is not being commanded by the
    # policy, and measures the same machine again; one blocking query for a fresh chunk is
    # all a new episode needs.
    policy = _FakePolicy(0.02)
    broker = rtc.RTCActionBroker(
        policy, rtc.RTCBrokerConfig(control_frequency=20.0, warmup_steps=2, calibration_steps=3)
    )
    broker.infer({"observation": 0})
    assert policy.calls == 5
    delay, horizon = broker.inference_delay, broker.execution_horizon

    broker.reset()
    broker.infer({"observation": 1})
    assert policy.calls == 6, "a new episode should cost one query, not another warm start"
    assert (broker.inference_delay, broker.execution_horizon) == (delay, horizon)
    # The chunk is fresh, and nothing of the previous episode is carried over.
    assert policy.requests[-1][rtc.RTC_PREV_CHUNK_ID] == 0


def test_requires_an_rtc_enabled_server():
    class _PlainPolicy:
        def infer(self, obs):
            return {"actions": np.zeros((ACTION_HORIZON, 4))}

        def reset(self):
            pass

    broker = rtc.RTCActionBroker(_PlainPolicy(), rtc.RTCBrokerConfig(control_frequency=20.0, calibration_steps=1))
    with pytest.raises(RuntimeError, match="rtc.enabled"):
        broker.infer({"observation": 0})


def test_config_requires_a_delay_source():
    with pytest.raises(ValueError, match="calibration_steps"):
        rtc.RTCBrokerConfig(control_frequency=20.0, calibration_steps=0)
