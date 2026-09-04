#!/usr/bin/env python3
"""Deploy a bimanual TeleAvatar V2 end-effector OpenPI policy."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys
import threading
import time

import numpy as np
from openpi_client import rtc as _rtc
from openpi_client import websocket_client_policy
import rclpy
import tyro

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.teleavatar_v2_ee.ros2_interface import TeleavatarV2EEInterface  # noqa: E402
from examples.teleavatar_v2_ee.ros2_interface import _rot6d_to_quaternion  # noqa: E402


@dataclasses.dataclass
class Args:
    remote_host: str = "127.0.0.1"
    """Policy server IP address."""

    remote_port: int = 8000
    """Policy server WebSocket port."""

    prompt: str = "perform the manipulation task"
    """Language instruction passed to the policy."""

    control_frequency: float = 45.0
    """Rate at which model waypoints are submitted as interpolation targets."""

    interp_frequency: float = 200.0
    """Rate at which interpolated Pose commands are published."""

    interpolate: bool = True
    """Interpolate each new target from the last Pose actually published."""

    open_loop_horizon: int = 30
    """Actions executed from each model chunk when RTC is disabled."""

    rtc: bool = False
    """Use Real-Time Chunking: infer in a background thread and return one action per tick."""

    rtc_warmup_steps: int = 2
    """RTC warm start inferences discarded before latency calibration."""

    rtc_calibration_steps: int = 5
    """RTC inferences timed at startup to estimate the end-to-end delay."""

    rtc_inference_delay: int | None = None
    """Override the measured RTC inference delay in control steps."""

    rtc_execution_horizon: int | None = None
    """Override the RTC execution horizon in control steps; default is twice the delay."""

    rtc_max_wait_s: float = 5.0
    """Maximum wait if the RTC action queue runs dry."""

    max_steps: int = 0
    """Stop after this many published/dry-run actions; 0 runs until Ctrl+C."""

    rtp_port: int = 8890
    rtp_payload: int = 96
    rtp_decoder: str = "nvh265dec max-display-delay=0"
    sensor_timeout: float = 1.0
    initial_data_timeout: float = 30.0

    initial_left_gripper_trigger: float = 0.0
    """Actual left gripper startup state: 0=open, 1=closed."""

    initial_right_gripper_trigger: float = 0.0
    """Actual right gripper startup state: 0=open, 1=closed."""

    publish: bool = False
    """Actually publish /api commands. Without --publish, inference is read-only."""

    max_position_error: float = 0.20
    """Stop if a target is farther from measured EE position than this (m); 0 disables."""

    max_orientation_error: float = 0.80
    """Stop if target/current EE orientation differs by more than this (rad); 0 disables."""


def _validate_args(args: Args) -> None:
    if args.control_frequency <= 0.0:
        raise ValueError("control_frequency must be positive")
    if args.interp_frequency <= 0.0:
        raise ValueError("interp_frequency must be positive")
    if args.open_loop_horizon <= 0:
        raise ValueError("open_loop_horizon must be positive")
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
        if args.rtc_max_wait_s <= 0.0:
            raise ValueError("rtc_max_wait_s must be positive")
    if args.max_steps < 0:
        raise ValueError("max_steps cannot be negative")
    if args.max_position_error < 0.0:
        raise ValueError("max_position_error cannot be negative")
    if args.max_orientation_error < 0.0:
        raise ValueError("max_orientation_error cannot be negative")
    for name in ("initial_left_gripper_trigger", "initial_right_gripper_trigger"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


def _validate_server_metadata(metadata: dict) -> None:
    expected = {
        "action_space": "bimanual_end_effector",
        "action_dim": 20,
    }
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual is None:
            logging.warning("Policy metadata has no %r field; cannot verify %s", key, expected_value)
        elif actual != expected_value:
            raise RuntimeError(f"Policy/client mismatch: metadata[{key!r}]={actual!r}, expected {expected_value!r}")


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


def _to_quaternion_action_chunk(actions: np.ndarray) -> np.ndarray:
    """Convert bimanual rot6d waypoints to quaternion waypoints once."""
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 20:
        raise ValueError(f"Expected action chunk with shape (N, 20), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Action chunk contains non-finite values")

    quaternion_actions = np.empty((len(values), 16), dtype=np.float32)
    for source_start, target_start in ((0, 0), (10, 8)):
        quaternion_actions[:, target_start : target_start + 3] = values[:, source_start : source_start + 3]
        source_quaternions = np.stack(
            [_rot6d_to_quaternion(rotation) for rotation in values[:, source_start + 3 : source_start + 9]]
        )
        for index in range(1, len(source_quaternions)):
            if float(np.dot(source_quaternions[index - 1], source_quaternions[index])) < 0.0:
                source_quaternions[index] *= -1.0
        quaternion_actions[:, target_start + 3 : target_start + 7] = source_quaternions
        quaternion_actions[:, target_start + 7] = values[:, source_start + 9]
    return quaternion_actions


def _to_quaternion_action(action: np.ndarray) -> np.ndarray:
    """Convert one 20D relative EE action to the 16D quaternion publish format."""
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (20,):
        raise ValueError(f"Expected one action with shape (20,), got {values.shape}")
    return _to_quaternion_action_chunk(values[None, ...])[0]


def _validate_and_publish_action(
    args: Args,
    interface: TeleavatarV2EEInterface,
    action: np.ndarray,
    *,
    playback_context: str,
) -> None:
    """Recheck live inputs and enforce EE safety immediately before publishing."""
    errors = interface.sensor_errors()
    if errors:
        raise RuntimeError(f"Inputs became unavailable during {playback_context}: " + "; ".join(errors))

    target_errors = interface.ee_quaternion_target_errors(
        action,
        max_position_error=args.max_position_error,
        max_orientation_error=args.max_orientation_error,
    )
    if target_errors:
        message = "EE target safety limit exceeded: " + "; ".join(target_errors)
        if args.publish:
            raise RuntimeError(message)
        logging.warning("Read-only safety warning: %s", message)

    if args.publish:
        interface.publish_quaternion_action(action)


def _run(args: Args, interface: TeleavatarV2EEInterface) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(host=args.remote_host, port=args.remote_port)
    metadata = client.get_server_metadata()
    logging.info("Connected to policy server; metadata=%s", metadata)
    _validate_server_metadata(metadata)

    if args.publish:
        logging.info(
            "Policy output enabled automatically after sensor/server checks at %.1f Hz",
            args.control_frequency,
        )
    else:
        logging.warning("Read-only mode: policy inference runs, but no /api commands are published")

    if args.rtc:
        _run_rtc(args, interface, client)
        return

    period = 1.0 / args.control_frequency
    total_steps = 0
    while args.max_steps == 0 or total_steps < args.max_steps:
        observation = interface.get_policy_observation(args.prompt)
        inference_started = time.monotonic()
        result = client.infer(observation)
        inference_ms = (time.monotonic() - inference_started) * 1000.0
        model_actions = _extract_action_chunk(result, args.open_loop_horizon)
        actions = _to_quaternion_action_chunk(model_actions)
        if args.max_steps:
            actions = actions[: args.max_steps - total_steps]
        if len(actions) == 0:
            break

        logging.info(
            "Received %d bimanual EE targets in %.1f ms; first left xyz=(%.3f, %.3f, %.3f), "
            "left trigger=%.3f, right xyz=(%.3f, %.3f, %.3f), right trigger=%.3f",
            len(actions),
            inference_ms,
            *actions[0, :3],
            actions[0, 7],
            *actions[0, 8:11],
            actions[0, 15],
        )

        next_target_at: float | None = None
        for action in actions:
            if next_target_at is not None:
                delay = next_target_at - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                # Match the mature runtime's pacing: a late step starts from
                # the current time instead of replaying missed deadlines.

            _validate_and_publish_action(
                args,
                interface,
                action,
                playback_context="action playback",
            )
            total_steps += 1
            # Anchor the next period to this completed submission. A delayed
            # step therefore slows the stream instead of releasing queued
            # waypoints in a burst to catch up with an obsolete deadline.
            next_target_at = time.monotonic() + period


def _run_rtc(
    args: Args,
    interface: TeleavatarV2EEInterface,
    client: websocket_client_policy.WebsocketClientPolicy,
) -> None:
    """Run the EE control loop one action at a time through the asynchronous RTC broker."""
    broker = _rtc.RTCActionBroker(
        policy=client,
        config=_rtc.RTCBrokerConfig(
            control_frequency=args.control_frequency,
            warmup_steps=args.rtc_warmup_steps,
            calibration_steps=args.rtc_calibration_steps,
            inference_delay=args.rtc_inference_delay,
            execution_horizon=args.rtc_execution_horizon,
            max_wait_s=args.rtc_max_wait_s,
        ),
    )

    period = 1.0 / args.control_frequency
    total_steps = 0
    next_target_at: float | None = None
    logging.info(
        "RTC EE control loop enabled: one action per %.1f Hz tick; "
        "server must be started with --rtc.enabled (use GUIDED for validation, TRAINED after RTC fine-tuning)",
        args.control_frequency,
    )
    try:
        while args.max_steps == 0 or total_steps < args.max_steps:
            if next_target_at is not None:
                delay = next_target_at - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)

            observation = interface.get_policy_observation(args.prompt)
            inference_started = time.monotonic()
            result = broker.infer(observation)
            inference_ms = (time.monotonic() - inference_started) * 1000.0
            raw_action = np.asarray(result.get("actions"), dtype=np.float32)
            if raw_action.shape != (20,):
                raise RuntimeError(f"RTC broker returned one action with shape (20,), got {raw_action.shape}")
            action = _to_quaternion_action(raw_action)

            _validate_and_publish_action(
                args,
                interface,
                action,
                playback_context="RTC playback",
            )
            total_steps += 1

            if total_steps == 1 or total_steps % max(int(args.control_frequency), 1) == 0:
                logging.info(
                    "RTC action %d: infer %.1f ms, left xyz=(%.3f, %.3f, %.3f), right xyz=(%.3f, %.3f, %.3f)",
                    total_steps,
                    inference_ms,
                    *action[:3],
                    *action[8:11],
                )
            next_target_at = time.monotonic() + period
    finally:
        broker.reset()


def main(args: Args) -> int:
    _validate_args(args)
    logging.info(
        "TeleAvatar V2 EE deployment: server=ws://%s:%d targets=%.1fHz interp_publish=%.1fHz "
        "horizon=%d rtc=%s interpolate=%s publish=%s",
        args.remote_host,
        args.remote_port,
        args.control_frequency,
        args.interp_frequency,
        args.open_loop_horizon,
        args.rtc,
        args.interpolate,
        args.publish,
    )
    logging.info(
        "EE safety limits: position=%.3fm orientation=%.3frad (0 disables each check)",
        args.max_position_error,
        args.max_orientation_error,
    )
    logging.info(
        "Startup gripper triggers are left=%.3f right=%.3f (0=open, 1=closed); they must match the physical grippers",
        args.initial_left_gripper_trigger,
        args.initial_right_gripper_trigger,
    )

    rclpy.init()
    interface: TeleavatarV2EEInterface | None = None
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    spin_thread: threading.Thread | None = None
    try:
        interface = TeleavatarV2EEInterface(
            rtp_port=args.rtp_port,
            rtp_payload=args.rtp_payload,
            rtp_decoder=args.rtp_decoder,
            sensor_timeout=args.sensor_timeout,
            control_frequency=args.control_frequency,
            interp_frequency=args.interp_frequency,
            interpolate=args.interpolate,
            initial_left_gripper_trigger=args.initial_left_gripper_trigger,
            initial_right_gripper_trigger=args.initial_right_gripper_trigger,
        )
        executor.add_node(interface)
        spin_thread = threading.Thread(target=executor.spin, name="teleavatar-ee-ros2", daemon=True)
        spin_thread.start()
        if not interface.wait_for_initial_data(args.initial_data_timeout):
            return 1
        _run(args, interface)
    except KeyboardInterrupt:
        logging.info("Ctrl+C received; stopping")
    finally:
        if interface is not None:
            interface.disable_output()
            time.sleep(0.05)
        executor.shutdown(timeout_sec=2.0)
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        if interface is not None:
            interface.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    raise SystemExit(main(tyro.cli(Args)))
