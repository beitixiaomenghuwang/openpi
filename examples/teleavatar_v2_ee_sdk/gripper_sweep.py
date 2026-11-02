#!/usr/bin/env python3
"""Test normalized gripper force commands while holding both current EE poses."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.teleavatar_v2_ee_sdk.sdk_interface import _CONTROL_ACTION_SEND_DURATION  # noqa: E402
from examples.teleavatar_v2_ee_sdk.sdk_interface import _load_sdk  # noqa: E402
from examples.teleavatar_v2_ee_sdk.sdk_interface import _pose_array  # noqa: E402
from examples.teleavatar_v2_ee_sdk.sdk_interface import _state_errors  # noqa: E402

# Same convention as this deployment's README and SDK 03_cartesian_gripper.cpp.
NEUTRAL_EFFORT = 0.10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--sdk-root", default="", help="Defaults to the sibling Teleavatar2 checkout")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--values", type=float, nargs="+", default=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 0])
    parser.add_argument("--hold", type=float, default=2.0, help="Seconds per value, or per transition with --ramp")
    parser.add_argument(
        "--hz", type=float, default=20.0, help="Python target update rate (>=10); native packets request 100 Hz"
    )
    parser.add_argument(
        "--ramp", action="store_true", help="Linearly interpolate between --values; hold the first value"
    )
    parser.add_argument("--publish", action="store_true", help="Connect, enable both arms, and run the test")
    args = parser.parse_args()
    if not args.robot_ip.strip():
        parser.error("--robot-ip must not be empty")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in args.values):
        parser.error("--values must all be finite numbers in [0, 1]")
    for name in ("hold", "hz"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive and finite")
    if args.hz < 10:
        parser.error("--hz must be at least 10 to check robot feedback at least every 0.1 seconds")
    return args


def checked_state(robot, *, controlling=False):
    snapshot = robot.state()
    errors = _state_errors(snapshot, timeout=1.0, controlling=controlling)
    if errors:
        raise RuntimeError("; ".join(errors))
    return snapshot


def wait_for_state(robot):
    deadline = time.monotonic() + 10.0
    errors = ["No SDK state received"]
    while time.monotonic() < deadline:
        snapshot = robot.state()
        errors = _state_errors(snapshot, timeout=1.0)
        if not errors:
            return snapshot
        if any("connection/session" in error or "Robot fault" in error for error in errors):
            raise RuntimeError("; ".join(errors))
        time.sleep(0.05)
    raise RuntimeError("Timed out waiting for SDK feedback: " + "; ".join(errors))


def transition(robot, action, wanted, timeout):
    checked_state(robot)
    robot.request_control_action(action)
    sent_at = time.monotonic()
    while time.monotonic() < sent_at + timeout:
        fsm = checked_state(robot)["frame"]["status"]["fsm"]
        if fsm == wanted and time.monotonic() >= sent_at + _CONTROL_ACTION_SEND_DURATION:
            return
        time.sleep(0.05)
    raise RuntimeError(f"{action}: expected {wanted}, got {fsm}; SDK error={robot.last_error!r}")


def gripper_feedback(snapshot, side):
    frame = snapshot["frame"]
    feedback = frame["feedback"]
    index = 4 if side == "left" else 5
    bit = 1 << index
    ages = feedback["source_age_us"]
    if (
        not feedback["received_mask"] & feedback["fresh_mask"] & bit
        or index >= len(ages)
        or ages[index] / 1e6 + snapshot["recv_age_s"] > min(1.0, feedback["stale_after_us"] / 1e6)
    ):
        return f"{side}: feedback missing/stale"
    for sample in reversed(frame.get("samples", [])):
        if (gripper := sample.get(f"{side}_gripper")) is not None:
            return f"{side}: angle={gripper['position']:.4f} rad, torque={gripper['effort']:.4f} Nm"
    return f"{side}: no gripper sample"


def sweep(robot, args):
    previous = args.values[0]
    next_log = 0.0
    for target in args.values:
        started = time.monotonic()
        while True:
            snapshot = checked_state(robot, controlling=True)
            now = time.monotonic()
            fraction = min((now - started) / args.hold, 1.0)
            value = previous + (target - previous) * fraction if args.ramp else target
            left = value if args.side in ("left", "both") else NEUTRAL_EFFORT
            right = value if args.side in ("right", "both") else NEUTRAL_EFFORT
            robot.set_gripper_efforts(left, right)
            if now >= next_log or fraction == 1.0:
                logging.info(
                    "cmd L=%.3f R=%.3f | %s | %s",
                    left,
                    right,
                    gripper_feedback(snapshot, "left"),
                    gripper_feedback(snapshot, "right"),
                )
                next_log = now + 0.25
            if fraction == 1.0:
                break
            time.sleep(min(1.0 / args.hz, args.hold * (1.0 - fraction)))
        previous = target
    # Let the final endpoint reach the native sender before replacing it.
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        checked_state(robot, controlling=True)
        time.sleep(0.02)


def stop_output(robot):
    # STOP also preempts an unfinished START/CONFIRM. Try every cleanup operation.
    try:
        for operation in (
            lambda: robot.set_enabled(False),
            robot.clear_grippers,
            lambda: robot.set_engage(False),
            lambda: robot.request_control_action("STOP"),
        ):
            try:
                operation()
            except Exception:
                logging.exception("SDK stop operation failed")
        time.sleep(0.2)
    finally:
        robot.stop_command_loop()


def run(robot, args):
    motion_started = False
    try:
        robot.connect(args.robot_ip, enable_composite_video=False, desired_cmd_rate_hz=100)
        frame = wait_for_state(robot)["frame"]
        left, right = (_pose_array(frame["ee"][f"{side}_current"]) for side in ("left", "right"))
        robot.set_pose_targets(left, right)
        motion_started = True
        robot.set_enabled(True)
        robot.set_engage(True)
        robot.start_command_loop()
        time.sleep(0.3)
        transition(robot, "START", "FSM_STATE_SLOW_START", 5.0)
        transition(robot, "CONFIRM", "FSM_STATE_READY", 10.0)
        logging.info("READY; holding both initial EE poses")
        sweep(robot, args)
        robot.set_gripper_efforts(NEUTRAL_EFFORT, NEUTRAL_EFFORT)
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            checked_state(robot, controlling=True)
            time.sleep(0.02)
    finally:
        try:
            if motion_started:
                stop_output(robot)
        finally:
            robot.disconnect()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("side=%s, values=%s, hold=%.2fs, ramp=%s", args.side, args.values, args.hold, args.ramp)
    if not args.publish:
        logging.info("Preview only; no robot connection. Add --publish to execute.")
        return
    logging.info("Both arms will be enabled at their current poses; any unselected gripper receives 0.10")
    try:
        run(_load_sdk(args.sdk_root).RobotClient(), args)
    except KeyboardInterrupt:
        logging.info("Interrupted; SDK shutdown attempted")


if __name__ == "__main__":
    main()
