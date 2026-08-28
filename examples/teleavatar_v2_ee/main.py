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
from openpi_client import websocket_client_policy
import rclpy
import tyro

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.teleavatar_v2_ee.ros2_interface import TeleavatarV2EEInterface  # noqa: E402


@dataclasses.dataclass
class Args:
    remote_host: str = "127.0.0.1"
    """Policy server IP address."""

    remote_port: int = 8000
    """Policy server WebSocket port."""

    prompt: str = "perform the manipulation task"
    """Language instruction passed to the policy."""

    control_frequency: float = 45.0
    """Action playback rate. The default matches the converted dataset FPS."""

    open_loop_horizon: int = 30
    """Actions executed from each model chunk (maximum 30 for the current config)."""

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

    wait_for_enter: bool = True
    """Require Enter immediately before enabling real command publication."""


def _validate_args(args: Args) -> None:
    if args.control_frequency <= 0.0:
        raise ValueError("control_frequency must be positive")
    if args.open_loop_horizon <= 0:
        raise ValueError("open_loop_horizon must be positive")
    if args.max_steps < 0:
        raise ValueError("max_steps cannot be negative")
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
            raise RuntimeError(
                f"Policy/client mismatch: metadata[{key!r}]={actual!r}, expected {expected_value!r}"
            )

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


def _run(args: Args, interface: TeleavatarV2EEInterface) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(host=args.remote_host, port=args.remote_port)
    metadata = client.get_server_metadata()
    logging.info("Connected to policy server; metadata=%s", metadata)
    _validate_server_metadata(metadata)

    if args.publish and args.wait_for_enter:
        input(
            f"Ready to control both arms at {args.control_frequency:.1f} Hz. "
            "Press Enter to enable policy output, or Ctrl+C to abort: "
        )
    elif not args.publish:
        logging.warning("Read-only mode: policy inference runs, but no /api commands are published")

    period = 1.0 / args.control_frequency
    total_steps = 0
    while args.max_steps == 0 or total_steps < args.max_steps:
        observation = interface.get_policy_observation(args.prompt)
        inference_started = time.monotonic()
        result = client.infer(observation)
        inference_ms = (time.monotonic() - inference_started) * 1000.0
        actions = _extract_action_chunk(result, args.open_loop_horizon)
        if args.max_steps:
            actions = actions[: args.max_steps - total_steps]

        logging.info(
            "Received %d bimanual EE actions in %.1f ms; first left xyz=(%.3f, %.3f, %.3f), left trigger=%.3f, right xyz=(%.3f, %.3f, %.3f), right trigger=%.3f",
            len(actions),
            inference_ms,
            *actions[0, :3],
            actions[0, 9],
            *actions[0, 10:13],
            actions[0, 19],
        )

        deadline = time.monotonic()
        for action in actions:
            errors = interface.sensor_errors()
            if errors:
                raise RuntimeError("Inputs became unavailable during action playback: " + "; ".join(errors))
            if args.publish:
                interface.publish_action(action)
            total_steps += 1
            deadline += period
            delay = deadline - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                logging.warning("Control loop missed deadline by %.1f ms", -delay * 1000.0)


def main(args: Args) -> int:
    _validate_args(args)
    logging.info(
        "TeleAvatar V2 EE deployment: server=ws://%s:%d control=%.1fHz horizon=%d publish=%s",
        args.remote_host,
        args.remote_port,
        args.control_frequency,
        args.open_loop_horizon,
        args.publish,
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
