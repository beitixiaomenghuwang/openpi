#!/usr/bin/env python3
"""Deploy a bimanual TeleAvatar V2 EE OpenPI policy through the C++ SDK."""

from __future__ import annotations

import dataclasses
import functools
import logging
import pathlib
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openpi_client import websocket_client_policy

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.teleavatar_v2_ee.se3_trajectory import blend_chunk_start  # noqa: E402
from examples.teleavatar_v2_ee_sdk.sdk_interface import _CAMERA_VIEWS  # noqa: E402
from examples.teleavatar_v2_ee_sdk.sdk_interface import SDKConfig  # noqa: E402
from examples.teleavatar_v2_ee_sdk.sdk_interface import TeleavatarV2EESDKInterface  # noqa: E402


@dataclasses.dataclass
class Args(SDKConfig):
    remote_host: str = "127.0.0.1"
    """Policy server IP address."""

    remote_port: int = 8000
    """Policy server WebSocket port."""

    prompt: str = "perform the manipulation task"
    """Language instruction passed to the policy."""

    open_loop_horizon: int = 30
    """Actions executed from each model chunk when RTC is disabled."""

    rtc: bool = False
    """Use Real-Time Chunking with background inference and one action per tick."""

    rtc_warmup_steps: int = 2
    """RTC warm start inferences discarded before latency calibration."""

    rtc_calibration_steps: int = 5
    """RTC inferences timed at startup to estimate the end-to-end delay."""

    rtc_inference_delay: int | None = None
    """Override the measured RTC inference delay in control steps."""

    rtc_execution_horizon: int | None = None
    """Override the RTC execution horizon; default is twice the delay."""

    rtc_max_wait_s: float = 5.0
    """Maximum wait if the RTC action queue runs dry."""

    chunk_blend_steps: int = 10
    """Without RTC: start each chunk at the previous last target and fade that offset out over
    this many targets. New chunks are anchored on the lagging measured pose; 0 disables."""

    rtc_blend_steps: int = 5
    """With RTC: cross-fade old/new chunk actions for this many steps after a switch; 0 disables."""

    max_steps: int = 0
    """Stop after this many submitted/dry-run actions; 0 runs until Ctrl+C."""


def _validate_args(args: Args) -> None:
    if args.open_loop_horizon <= 0:
        raise ValueError("open_loop_horizon must be positive")
    for name in ("chunk_blend_steps", "rtc_blend_steps"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} cannot be negative")
    if args.rtc:
        if args.rtc_warmup_steps < 0:
            raise ValueError("rtc_warmup_steps cannot be negative")
        if args.rtc_calibration_steps < 0:
            raise ValueError("rtc_calibration_steps cannot be negative")
        if args.rtc_calibration_steps == 0 and args.rtc_inference_delay is None:
            raise ValueError("rtc_calibration_steps=0 requires rtc_inference_delay")
        if args.rtc_inference_delay is not None and args.rtc_inference_delay < 0:
            raise ValueError("rtc_inference_delay cannot be negative")
        if args.rtc_execution_horizon is not None and args.rtc_execution_horizon <= 0:
            raise ValueError("rtc_execution_horizon must be positive")
        if not np.isfinite(args.rtc_max_wait_s) or args.rtc_max_wait_s <= 0.0:
            raise ValueError("rtc_max_wait_s must be finite and positive")
    if args.max_steps < 0:
        raise ValueError("max_steps cannot be negative")


def _validate_server_metadata(metadata: dict) -> None:
    expected = {"action_space": "bimanual_end_effector", "action_dim": 20}
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual is None:
            logging.warning("Policy metadata has no %r field; cannot verify %s", key, expected_value)
        elif actual != expected_value:
            raise RuntimeError(f"Policy/client mismatch: metadata[{key!r}]={actual!r}, expected {expected_value!r}")
    cameras = metadata.get("cameras")
    if cameras is not None:
        unknown = set(cameras) - _CAMERA_VIEWS.keys()
        if unknown:
            raise RuntimeError(f"Policy metadata lists cameras this client does not provide: {sorted(unknown)}")


def _extract_action_chunk(result: dict, horizon: int) -> np.ndarray:
    if "actions" not in result:
        raise RuntimeError(f"Policy response has no 'actions' field: {result.keys()}")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 20:
        raise RuntimeError(f"Expected policy actions with shape (N, 20), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Policy returned non-finite actions")
    if horizon > len(actions):
        raise RuntimeError(f"open_loop_horizon={horizon} exceeds returned chunk length {len(actions)}")
    return actions[:horizon]


def _rot6d_to_quaternion(rot6d: np.ndarray) -> np.ndarray:
    """Convert the policy's first two rotation-matrix rows to an xyzw quaternion."""
    value = np.array(rot6d, dtype=np.float64, copy=True)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Expected finite rotation-6D shape (6,), got {value.shape}")
    row_1 = value[:3]
    row_2 = value[3:]
    norm_1 = float(np.linalg.norm(row_1))
    if norm_1 < 1e-7:
        raise ValueError("Rotation-6D first row is degenerate")
    row_1 /= norm_1
    row_2 -= np.dot(row_1, row_2) * row_1
    norm_2 = float(np.linalg.norm(row_2))
    if norm_2 < 1e-7:
        raise ValueError("Rotation-6D second row is degenerate")
    row_2 /= norm_2
    return _matrix_to_quaternion(np.stack((row_1, row_2, np.cross(row_1, row_2)), axis=0))


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an xyzw quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (m[2, 1] - m[1, 2]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
                (m[1, 0] - m[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0))
            quaternion = np.array(
                [0.25 * scale, (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale, (m[2, 1] - m[1, 2]) / scale]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0))
            quaternion = np.array(
                [(m[0, 1] + m[1, 0]) / scale, 0.25 * scale, (m[1, 2] + m[2, 1]) / scale, (m[0, 2] - m[2, 0]) / scale]
            )
        else:
            scale = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0))
            quaternion = np.array(
                [(m[0, 2] + m[2, 0]) / scale, (m[1, 2] + m[2, 1]) / scale, 0.25 * scale, (m[1, 0] - m[0, 1]) / scale]
            )
    return (quaternion / np.linalg.norm(quaternion)).astype(np.float32)


def _to_quaternion_action_chunk(actions: np.ndarray) -> np.ndarray:
    """Change rotation representation without composing poses again.

    TeleavatarEEOutputs on the policy server already returns absolute EE poses,
    even when the model's action_representation metadata describes relative EE.
    The SDK interface handles the normalized gripper entries.
    """
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 20 or len(values) == 0:
        raise ValueError(f"Expected nonempty action chunk with shape (N, 20), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Action chunk contains non-finite values")

    quaternion_actions = np.empty((len(values), 16), dtype=np.float32)
    for source_start, target_start in ((0, 0), (10, 8)):
        quaternion_actions[:, target_start : target_start + 3] = values[:, source_start : source_start + 3]
        quaternions = np.stack(
            [_rot6d_to_quaternion(rotation) for rotation in values[:, source_start + 3 : source_start + 9]]
        )
        for index in range(1, len(quaternions)):
            if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
                quaternions[index] *= -1.0
        quaternion_actions[:, target_start + 3 : target_start + 7] = quaternions
        quaternion_actions[:, target_start + 7] = values[:, source_start + 9]
    return quaternion_actions


def _to_quaternion_action(action: np.ndarray) -> np.ndarray:
    """Convert one absolute 20D EE action to the 16D xyzw SDK-interface format."""
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (20,):
        raise ValueError(f"Expected one action with shape (20,), got {values.shape}")
    return _to_quaternion_action_chunk(values[None, ...])[0]


def _next_deadline(previous: float | None, period: float) -> float:
    """Keep a fixed target cadence so per-step overhead does not stretch every period.

    A step that finished a full period late restarts the cadence instead of releasing
    queued targets in a burst to catch up with an obsolete deadline.
    """
    now = time.monotonic()
    if previous is None or now - previous >= period:
        return now + period
    return previous + period


def _validate_and_publish_action(
    args: Args,
    interface: TeleavatarV2EESDKInterface,
    action: np.ndarray,
    *,
    playback_context: str,
) -> None:
    """Recheck live inputs immediately before submitting each target."""
    errors = interface.sensor_errors()
    if errors:
        raise RuntimeError(f"Inputs became unavailable during {playback_context}: " + "; ".join(errors))
    if args.publish:
        interface.publish_quaternion_action(action)


def _run(args: Args, interface: TeleavatarV2EESDKInterface) -> None:
    """Enter SDK current-pose hold before capturing the first policy observation."""
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(host=args.remote_host, port=args.remote_port)
    metadata = client.get_server_metadata()
    logging.info("Connected to policy server; metadata=%s", metadata)
    _validate_server_metadata(metadata)
    errors = interface.sensor_errors()
    if errors:
        raise RuntimeError("Inputs unavailable after connecting to the policy server: " + "; ".join(errors))
    if args.publish:
        interface.enable_output()
        logging.info("SDK ready in current-pose hold; policy targets will run at %.1f Hz", args.control_frequency)
    else:
        logging.warning("Read-only mode: policy inference runs without enabling robot control")

    if args.rtc:
        _run_rtc(args, interface, client)
        return

    period = 1.0 / args.control_frequency
    total_steps = 0
    previous_last: np.ndarray | None = None
    while args.max_steps == 0 or total_steps < args.max_steps:
        observation = interface.get_policy_observation(args.prompt)
        inference_started = time.monotonic()
        result = client.infer(observation)
        inference_ms = (time.monotonic() - inference_started) * 1000.0
        actions = _to_quaternion_action_chunk(_extract_action_chunk(result, args.open_loop_horizon))
        if args.max_steps:
            actions = actions[: args.max_steps - total_steps]
        logging.info(
            "Received %d bimanual EE targets in %.1f ms; first left xyz=(%.3f, %.3f, %.3f), "
            "left gripper=%.3f, right xyz=(%.3f, %.3f, %.3f), right gripper=%.3f",
            len(actions),
            inference_ms,
            *actions[0, :3],
            actions[0, 7],
            *actions[0, 8:11],
            actions[0, 15],
        )
        if previous_last is not None:
            actions = blend_chunk_start(previous_last, actions, args.chunk_blend_steps)
        previous_last = actions[-1].copy()
        next_target_at: float | None = None
        for action in actions:
            if next_target_at is not None:
                delay = next_target_at - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
            _validate_and_publish_action(args, interface, action, playback_context="action playback")
            total_steps += 1
            next_target_at = _next_deadline(next_target_at, period)


def _run_rtc(
    args: Args,
    interface: TeleavatarV2EESDKInterface,
    client: websocket_client_policy.WebsocketClientPolicy,
) -> None:
    from openpi_client import rtc as _rtc

    broker = _rtc.RTCActionBroker(
        policy=client,
        config=_rtc.RTCBrokerConfig(
            control_frequency=args.control_frequency,
            warmup_steps=args.rtc_warmup_steps,
            calibration_steps=args.rtc_calibration_steps,
            inference_delay=args.rtc_inference_delay,
            execution_horizon=args.rtc_execution_horizon,
            max_wait_s=args.rtc_max_wait_s,
            blend_steps=args.rtc_blend_steps,
        ),
    )
    period = 1.0 / args.control_frequency
    total_steps = 0
    next_target_at: float | None = None
    # Evaluated by the broker only when it launches an inference request (on its worker
    # thread); playback ticks no longer read state and copy images.
    observation = functools.partial(interface.get_policy_observation, args.prompt)
    logging.info(
        "RTC EE control loop enabled: one action per %.1f Hz tick; server must be started with --rtc.enabled",
        args.control_frequency,
    )
    try:
        while args.max_steps == 0 or total_steps < args.max_steps:
            if next_target_at is not None:
                delay = next_target_at - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
            tick_started = time.monotonic()
            result = broker.infer(observation)
            wait_ms = (time.monotonic() - tick_started) * 1000.0
            raw_action = np.asarray(result.get("actions"), dtype=np.float32)
            if raw_action.shape != (20,):
                raise RuntimeError(f"RTC broker expected one action with shape (20,), got {raw_action.shape}")
            action = _to_quaternion_action(raw_action)
            _validate_and_publish_action(args, interface, action, playback_context="RTC playback")
            total_steps += 1
            if total_steps == 1 or total_steps % max(int(args.control_frequency), 1) == 0:
                logging.info(
                    "RTC action %d: broker wait %.1f ms, left xyz=(%.3f, %.3f, %.3f), right xyz=(%.3f, %.3f, %.3f)",
                    total_steps,
                    wait_ms,
                    *action[:3],
                    *action[8:11],
                )
            next_target_at = _next_deadline(next_target_at, period)
    finally:
        broker.reset()


def main(args: Args) -> int:
    _validate_args(args)
    logging.info(
        "TeleAvatar V2 SDK EE deployment: server=ws://%s:%d targets=%.1fHz sdk_commands=%.1fHz "
        "horizon=%d rtc=%s interpolate=%s publish=%s",
        args.remote_host,
        args.remote_port,
        args.control_frequency,
        args.interp_frequency if args.interpolate else args.control_frequency,
        args.open_loop_horizon,
        args.rtc,
        args.interpolate,
        args.publish,
    )
    interface = TeleavatarV2EESDKInterface(args)
    try:
        interface.connect()
        if not interface.wait_for_initial_data(args.initial_data_timeout):
            return 1
        _run(args, interface)
    except KeyboardInterrupt:
        logging.info("Ctrl+C received; stopping")
    finally:
        interface.close()
    return 0


if __name__ == "__main__":
    import tyro

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    raise SystemExit(main(tyro.cli(Args)))
