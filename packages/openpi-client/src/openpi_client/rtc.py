"""Client side of Real-Time Chunking.

Drop-in for `ActionChunkBroker`, but queries the policy from a background thread while the
control loop keeps draining the chunk it already has, telling the server how much of that
chunk will have been executed by the time the answer arrives. Needs a server started with
`--rtc.enabled`. See docs/real_time_chunking.md.
"""

import collections
import concurrent.futures
import dataclasses
import logging
import math
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy

logger = logging.getLogger(__name__)

# Must match openpi.policies.policy.
RTC_PREV_CHUNK_ID = "rtc/prev_chunk_id"
RTC_PREFIX_START = "rtc/prefix_start"
RTC_INFERENCE_DELAY = "rtc/inference_delay"
RTC_EXECUTION_HORIZON = "rtc/execution_horizon"
RTC_CHUNK_ID = "rtc/chunk_id"
RTC_ACTION_HORIZON = "rtc/action_horizon"


@dataclasses.dataclass
class RTCBrokerConfig:
    """How the delay and execution horizon are obtained: measured at startup, or pinned."""

    # Rate at which the control loop calls `infer()`; converts latency into control steps.
    control_frequency: float

    # Warm start. Two JAX compilations happen here (unconstrained, then constrained), so
    # warmup_steps should be >= 2 for the measurement to land on warm code.
    warmup_steps: int = 2
    calibration_steps: int = 5
    # 1.0 = worst case. Underestimating the delay leaves executed steps unconstrained.
    calibration_quantile: float = 1.0
    delay_margin_steps: int = 1

    # Explicit values, skipping the measurement for whichever is set.
    inference_delay: Optional[int] = None
    execution_horizon: Optional[int] = None
    # Used when execution_horizon is None: clip(scale * delay, min_execution_horizon, chunk).
    execution_horizon_scale: float = 2.0
    min_execution_horizon: int = 4

    # Keep tracking the real delay during the episode.
    adapt_delay: bool = True
    adapt_window: int = 20

    # How long `infer()` may block if the queue drains before a result arrives.
    max_wait_s: float = 5.0

    def __post_init__(self) -> None:
        if self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive")
        if self.calibration_steps == 0 and self.inference_delay is None:
            raise ValueError("Set either calibration_steps > 0 or an explicit inference_delay.")


class RTCActionBroker(_base_policy.BasePolicy):
    """Serves an action chunk one step at a time, refilling it asynchronously.

    Expects a single control-loop caller (what `Runtime` provides); the inference thread is
    the only other participant.
    """

    def __init__(self, policy: _base_policy.BasePolicy, config: RTCBrokerConfig):
        self._policy = policy
        self._config = config
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-infer")
        self._lock = threading.Lock()
        self._reset_state()

    def _reset_state(self, *, keep_calibration: bool = False) -> None:
        self._chunk: Optional[Dict[str, Any]] = None
        self._cursor = 0
        self._step = 0
        self._chunk_id = 0
        self._pending: Optional[concurrent.futures.Future] = None
        self._pending_step = 0
        self._last_request_step = 0
        self._calibrated = False
        if not keep_calibration:
            self._action_horizon = 0
            self._inference_delay = self._config.inference_delay or 0
            self._execution_horizon = self._config.execution_horizon or 0
            self._observed_delays: collections.deque = collections.deque(maxlen=self._config.adapt_window)
            self._measured = False

    @property
    def inference_delay(self) -> int:
        return self._inference_delay

    @property
    def execution_horizon(self) -> int:
        return self._execution_horizon

    def _query(self, obs: Dict, *, prev_chunk_id: int, prefix_start: int, delay: int, horizon: int) -> Dict:
        """One blocking round trip. Runs on the worker thread, so delay/horizon are
        snapshotted by the caller rather than read here."""
        request = dict(obs)
        request[RTC_PREV_CHUNK_ID] = int(prev_chunk_id)
        request[RTC_PREFIX_START] = int(prefix_start)
        request[RTC_INFERENCE_DELAY] = int(delay)
        request[RTC_EXECUTION_HORIZON] = int(horizon)
        result = self._policy.infer(request)
        if RTC_CHUNK_ID not in result:
            raise RuntimeError(
                "Policy server did not return an RTC chunk id. Start it with `--rtc.enabled` "
                "(scripts/serve_policy.py), or use ActionChunkBroker instead of RTCActionBroker."
            )
        return result

    def _install(self, result: Dict, cursor: int) -> None:
        self._chunk = result
        self._cursor = cursor
        self._chunk_id = int(result[RTC_CHUNK_ID])
        self._action_horizon = int(result.get(RTC_ACTION_HORIZON, len(result["actions"])))

    def _bootstrap(self, obs: Dict) -> None:
        """Size the delay from a few unconstrained inferences, then start executing.

        Runs inside the first `infer()` call, which therefore returns before the runtime
        ever reaches `apply_action`: nothing is commanded while measuring.
        """
        cfg = self._config
        if self._measured:
            # A later episode: the delay is a property of this machine, not of the episode,
            # and the server's graphs are already compiled. One blocking query is enough to
            # get a chunk to execute.
            total = 1
        else:
            total = cfg.warmup_steps + cfg.calibration_steps if cfg.calibration_steps else 1
        latencies = []
        result = None
        chunk_id = 0  # 0 on the first query only: nothing is executing yet
        horizon = self._execution_horizon
        for i in range(total):
            started = time.perf_counter()
            # Every query after the first carries a previous chunk, so the server traces and
            # compiles the *constrained* graph here rather than on the first real request --
            # and the latencies measured below are the ones steady state will actually see.
            result = self._query(
                obs,
                prev_chunk_id=chunk_id,
                prefix_start=0,
                delay=self._inference_delay,
                horizon=horizon,
            )
            chunk_id = int(result[RTC_CHUNK_ID])
            # Constrain over the whole chunk until calibration produces a real horizon; the
            # chunks all come from this same observation, so it costs nothing.
            horizon = int(result.get(RTC_ACTION_HORIZON, len(result["actions"])))
            elapsed = time.perf_counter() - started
            if cfg.calibration_steps and i >= cfg.warmup_steps:
                latencies.append(elapsed)
            if self._measured:
                logger.info("RTC re-arm: %.1f ms (reusing the measured delay)", elapsed * 1000)
            else:
                logger.info(
                    "RTC warm start %d/%d: %.1f ms%s",
                    i + 1,
                    total,
                    elapsed * 1000,
                    "" if latencies else " (discarded)",
                )

        assert result is not None
        self._action_horizon = int(result.get(RTC_ACTION_HORIZON, len(result["actions"])))
        if latencies:
            self._inference_delay = self._delay_from_latencies(latencies)
        self._execution_horizon = self._resolve_execution_horizon()
        self._calibrated = True
        if not self._measured:
            self._measured = True
            logger.info(
                "RTC calibrated: inference_delay=%d steps (%.0f ms @ %.1f Hz), execution_horizon=%d, chunk=%d",
                self._inference_delay,
                self._inference_delay / cfg.control_frequency * 1000,
                cfg.control_frequency,
                self._execution_horizon,
                self._action_horizon,
            )
        # The last warm-start chunk is still aligned with step 0: same observation, and the
        # robot has not been commanded since.
        self._install(result, cursor=0)
        self._last_request_step = self._step

    def _delay_from_steps(self, step_delays) -> int:
        cfg = self._config
        worst = float(np.quantile(np.asarray(step_delays, dtype=float), cfg.calibration_quantile))
        # The epsilon stops float noise (4.0000000001) rounding up a whole step.
        steps = math.ceil(worst - 1e-9) + cfg.delay_margin_steps
        # Leave one step unconstrained, otherwise the new chunk is fully pinned.
        return int(np.clip(steps, 0, max(self._action_horizon - 1, 0)))

    def _delay_from_latencies(self, latencies) -> int:
        return self._delay_from_steps([latency * self._config.control_frequency for latency in latencies])

    def _resolve_execution_horizon(self) -> int:
        cfg = self._config
        if cfg.execution_horizon is not None:
            horizon = cfg.execution_horizon
        else:
            horizon = max(round(cfg.execution_horizon_scale * self._inference_delay), cfg.min_execution_horizon)
        # Leave room for at least one step of progress per inference round.
        return int(np.clip(horizon, self._inference_delay + 1, self._action_horizon))

    def _collect(self, *, block: bool) -> None:
        """Install a finished inference, if there is one."""
        if self._pending is None:
            return
        if not block and not self._pending.done():
            return
        timeout = self._config.max_wait_s if block else None
        try:
            result = self._pending.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            raise RuntimeError(
                f"RTC: queue ran dry and inference did not finish within {self._config.max_wait_s}s; "
                "lower execution_horizon or control_frequency."
            ) from e
        finally:
            if self._pending is not None and self._pending.done():
                self._pending = None

        real_delay = self._step - self._pending_step
        self._observed_delays.append(real_delay)
        if real_delay > self._inference_delay:
            logger.warning(
                "RTC: inference took %d control steps but only %d were constrained; raise delay_margin_steps.",
                real_delay,
                self._inference_delay,
            )
        # The chunk is indexed from the step the request was issued, so index `real_delay`
        # is the step about to execute.
        self._install(result, cursor=min(real_delay, self._action_horizon))
        if self._config.adapt_delay and self._observed_delays:
            self._inference_delay = self._delay_from_steps(self._observed_delays)
            if self._config.execution_horizon is None:
                self._execution_horizon = self._resolve_execution_horizon()

    def _maybe_launch(self, obs: Dict) -> None:
        if self._pending is not None:
            return
        # Fire `inference_delay` steps before the execution horizon runs out, so the answer
        # lands on the boundary.
        elapsed = self._step - self._last_request_step
        if elapsed < max(self._execution_horizon - self._inference_delay, 1):
            return
        prev_chunk_id, prefix_start = self._chunk_id, self._cursor
        self._pending_step = self._step
        self._last_request_step = self._step
        self._pending = self._executor.submit(
            self._query,
            obs,
            prev_chunk_id=prev_chunk_id,
            prefix_start=prefix_start,
            delay=self._inference_delay,
            horizon=self._execution_horizon,
        )

    def _pop(self) -> Dict:
        if self._cursor >= len(self._chunk["actions"]):
            # Wait for the refill rather than repeating a stale command.
            logger.warning("RTC: action queue empty, blocking on in-flight inference")
            self._collect(block=True)
            if self._cursor >= len(self._chunk["actions"]):
                raise RuntimeError("RTC: action queue still empty after refill; execution_horizon is too large.")

        index = self._cursor
        chunk_len = len(self._chunk["actions"])

        def slicer(x):
            # Index the chunked fields only; scalars and metadata pass through.
            if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] == chunk_len:
                return x[index, ...]
            return x

        self._cursor += 1
        return tree.map_structure(slicer, self._chunk)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self._lock:
            if not self._calibrated:
                self._bootstrap(obs)
            else:
                self._step += 1
                self._collect(block=False)
                self._maybe_launch(obs)
            return self._pop()

    @override
    def reset(self) -> None:
        with self._lock:
            if self._pending is not None:
                self._pending.cancel()
                concurrent.futures.wait([self._pending], timeout=self._config.max_wait_s)
            self._policy.reset()
            # Keep the calibration: it measures this machine, not this episode.
            self._reset_state(keep_calibration=True)
