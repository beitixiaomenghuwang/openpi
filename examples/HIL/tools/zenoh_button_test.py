#!/usr/bin/env python3
"""Read-only live controller-input test for the second-S100 Zenoh XR stream.

The script subscribes to ``sec/xr/devicepose`` using Zenoh peer discovery,
decodes the input bitmask, both sticks, triggers, and squeezes, and prints
their changes. It never imports ROS and never publishes robot commands.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import threading
import time
from typing import Any

from hil.topics import VR_ZENOH_DEVICEPOSE_KEY
from hil.zenoh_wire import WireDecodeError, parse_wire_fields


# Verified from the controlled capture in /tmp/xr_mapping.jsonl.
BUTTON_MASK_FIELD = 6
BUTTON_BITS = {
    0x01: "left X",
    0x02: "left Y",
    0x04: "right A",
    0x08: "right B",
    0x10: "left trigger",
    0x20: "right trigger",
    0x40: "left squeeze",
    0x80: "right squeeze",
}

# Fields 8 and 9 are always-present pairs of float32 values. Their left/right
# assignment follows the surrounding left/right field pairs (10/11, 12/13,
# 16/17) and the legacy XR input layout. The live display makes this mapping
# easy to confirm by moving only one stick at a time.
STICK_FIELDS = {
    8: "left stick",
    9: "right stick",
}

ANALOG_FIELDS = {
    10: "left trigger",
    11: "right trigger",
    12: "left squeeze",
    13: "right squeeze",
}

# Pose fields are documented here so the decoded sec_dev mapping has one
# explicit reference, although this read-only input test does not use poses.
POSE_FIELDS = {
    14: "head pose",
    16: "left controller pose",
    17: "right controller pose",
    18: "left controller pose duplicate",
    19: "right controller pose duplicate",
}


def _payload_bytes(payload: Any) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if hasattr(payload, "to_bytes"):
        return payload.to_bytes()
    return bytes(payload)


def extract_controller_inputs(
    payload: bytes,
) -> tuple[int, dict[str, tuple[float, float]], dict[str, float]]:
    """Decode the mapped controller-input fields from one raw payload."""
    mask = 0
    sticks = {name: (0.0, 0.0) for name in STICK_FIELDS.values()}
    # Trigger/squeeze fields are optional in the wire message. An omitted
    # field means "no new value", so the live display must not reset it.
    analog: dict[str, float] = {}
    for field in parse_wire_fields(payload):
        if field.number == BUTTON_MASK_FIELD and field.wire_type == 0:
            mask |= int(field.value)
            continue

        stick_name = STICK_FIELDS.get(field.number)
        if stick_name is not None:
            if field.wire_type != 2 or len(field.raw) != 8:
                raise WireDecodeError(
                    f"field {field.number} ({stick_name}) must contain two float32 values"
                )
            x, y = struct.unpack("<ff", field.raw)
            if not math.isfinite(x) or not math.isfinite(y):
                raise WireDecodeError(f"field {field.number} contains a non-finite stick value")
            sticks[stick_name] = (float(x), float(y))
            continue

        analog_name = ANALOG_FIELDS.get(field.number)
        if analog_name is not None:
            if field.wire_type != 5 or not math.isfinite(float(field.value)):
                raise WireDecodeError(
                    f"field {field.number} ({analog_name}) must be a finite float32"
                )
            analog[analog_name] = float(field.value)
    return mask, sticks, analog


def extract_input_mask(payload: bytes) -> int:
    """Return the OR-ed field-6 input mask from one raw payload."""
    return extract_controller_inputs(payload)[0]


def extract_stick_values(payload: bytes) -> dict[str, tuple[float, float]]:
    """Return field-8/9 left and right stick values."""
    return extract_controller_inputs(payload)[1]


def extract_analog_values(payload: bytes) -> dict[str, float]:
    """Return only trigger/squeeze values present in the payload."""
    return extract_controller_inputs(payload)[2]


def _label_for_bit(bit: int) -> str:
    return BUTTON_BITS.get(bit, f"unknown bit 0x{bit:02x}")


def describe_edges(previous: int, current: int) -> list[str]:
    """Return human-readable rising/falling edges between two masks."""
    edges = []
    changed = previous ^ current
    bit = 1
    while bit <= 0x80000000:
        if changed & bit:
            state = "pressed" if current & bit else "released"
            edges.append(f"{state}: {_label_for_bit(bit)} (0x{bit:02x})")
        bit <<= 1
    return edges


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default=VR_ZENOH_DEVICEPOSE_KEY)
    parser.add_argument(
        "--endpoint", default="", help="Optional Zenoh endpoint, e.g. tcp/192.168.1.20:7447"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after this many seconds; zero means run until Ctrl+C",
    )
    parser.add_argument(
        "--show-analog",
        action="store_true",
        help="Print trigger/squeeze values when they change",
    )
    parser.add_argument(
        "--no-sticks",
        action="store_true",
        help="Do not print left/right stick movement",
    )
    parser.add_argument(
        "--stick-deadzone",
        type=float,
        default=0.08,
        help="Treat both stick axes as centered below this absolute value",
    )
    parser.add_argument(
        "--stick-delta",
        type=float,
        default=0.05,
        help="Minimum axis change before printing another stick value",
    )
    args = parser.parse_args(argv)
    if args.duration < 0.0:
        parser.error("--duration must be non-negative")
    if not 0.0 <= args.stick_deadzone < 1.0:
        parser.error("--stick-deadzone must be in [0, 1)")
    if not 0.0 < args.stick_delta <= 2.0:
        parser.error("--stick-delta must be in (0, 2]")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        import zenoh
    except ImportError as exc:
        print(
            "eclipse-zenoh is required: python3 -m pip install eclipse-zenoh",
            file=sys.stderr,
        )
        return 2

    config = zenoh.Config()
    if args.endpoint:
        config.insert_json5("connect/endpoints", json.dumps([args.endpoint]))

    state_lock = threading.Lock()
    previous_mask: int | None = None
    last_sticks: dict[str, tuple[float, float]] = {}
    last_analog: dict[str, float] = {}
    packets = 0
    rejected = 0
    started = time.monotonic()

    session = zenoh.open(config)
    print(
        f"[OK] zenoh session opened ({args.endpoint or 'peer, multicast auto-discovery'})"
    )
    print(f"[SUB] listening on '{args.key}'")
    print("[INFO] read-only mode: no ROS node and no robot command publisher")
    print("[MAP] f6=buttons f8=left-stick f9=right-stick f10/11=triggers f12/13=squeezes")
    print("[INFO] press buttons or move sticks on the VR controllers; Ctrl+C exits")

    def listener(sample) -> None:
        nonlocal previous_mask, last_sticks, last_analog, packets, rejected
        try:
            payload = _payload_bytes(sample.payload)
            current_mask, sticks, analog = extract_controller_inputs(payload)
        except (TypeError, ValueError, WireDecodeError) as exc:
            with state_lock:
                rejected += 1
            print(f"[DROP] invalid payload: {exc}", file=sys.stderr)
            return

        with state_lock:
            packets += 1
            if previous_mask is None:
                previous_mask = current_mask
                if current_mask:
                    for bit in BUTTON_BITS:
                        if current_mask & bit:
                            print(f"[STATE] pressed: {_label_for_bit(bit)} (0x{bit:02x})")
            else:
                for edge in describe_edges(previous_mask, current_mask):
                    print(f"[BUTTON] {edge}")
                previous_mask = current_mask

            if not args.no_sticks:
                for name, raw_value in sticks.items():
                    value = (
                        (0.0, 0.0)
                        if max(abs(raw_value[0]), abs(raw_value[1])) < args.stick_deadzone
                        else raw_value
                    )
                    old = last_sticks.get(name)
                    if old is None:
                        if value != (0.0, 0.0):
                            print(f"[STICK] {name}: x={value[0]:+.3f} y={value[1]:+.3f}")
                    elif (
                        (old == (0.0, 0.0)) != (value == (0.0, 0.0))
                        or max(abs(value[0] - old[0]), abs(value[1] - old[1]))
                        >= args.stick_delta
                    ):
                        print(f"[STICK] {name}: x={value[0]:+.3f} y={value[1]:+.3f}")
                    last_sticks[name] = value

            if args.show_analog:
                for name, value in sorted(analog.items()):
                    old = last_analog.get(name)
                    if old is None or abs(value - old) >= 1.0 / 255.0:
                        print(f"[ANALOG] {name}: {value:.3f}")
                    last_analog[name] = value

    subscriber = session.declare_subscriber(args.key, listener)
    try:
        while True:
            if args.duration and time.monotonic() - started >= args.duration:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[STOP] interrupted")
    finally:
        try:
            subscriber.undeclare()
        except Exception as exc:
            print(f"[WARN] subscriber cleanup failed: {exc}", file=sys.stderr)
        try:
            session.close()
        except Exception as exc:
            print(f"[WARN] Zenoh session cleanup failed: {exc}", file=sys.stderr)

    with state_lock:
        print(f"[SUMMARY] packets={packets} rejected={rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
