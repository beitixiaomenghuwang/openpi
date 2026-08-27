"""Decoder for the current second-S100 ``sec/xr/devicepose`` payload."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from .pose_math import as_vector, normalize_quaternion
    from .zenoh_wire import WireDecodeError, parse_wire_fields
except ImportError:
    from pose_math import as_vector, normalize_quaternion
    from zenoh_wire import WireDecodeError, parse_wire_fields


BUTTON_MASK_FIELD = 6
STICK_FIELDS = {8: "left", 9: "right"}
ANALOG_FIELD_BY_SIDE = {
    "left": {"trigger": 10, "squeeze": 12},
    "right": {"trigger": 11, "squeeze": 13},
}
ANALOG_MASK_BY_SIDE = {
    "left": {"trigger": 0x10, "squeeze": 0x40},
    "right": {"trigger": 0x20, "squeeze": 0x80},
}
POSE_FIELDS = {
    14: "head",
    16: "left",
    17: "right",
    18: "left",
    19: "right",
}
CONTROLLER_POSE_CANDIDATES = {"left": (16, 18), "right": (17, 19)}


@dataclass
class PoseValue:
    position: np.ndarray
    quaternion: np.ndarray


@dataclass
class DecodedSecDevPose:
    poses: dict[str, PoseValue]
    input_mask: int
    analog: dict[str, dict[str, float]]
    sticks: dict[str, tuple[float, float]]


def _decode_pose_field(field) -> PoseValue:
    if field.wire_type != 2 or len(field.raw) != 28:
        raise WireDecodeError(
            f"pose field {field.number} must be a 28-byte length-delimited value"
        )
    values = np.asarray(np.frombuffer(field.raw, dtype="<f4"), dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise WireDecodeError(f"pose field {field.number} contains non-finite values")
    return PoseValue(
        position=as_vector(values[:3], 3),
        quaternion=normalize_quaternion(values[3:]),
    )


def decode_sec_dev_pose(payload: bytes) -> DecodedSecDevPose:
    """Decode controller poses, button mask, sticks, and analog controls."""
    fields = parse_wire_fields(payload)
    poses_by_field: dict[int, PoseValue] = {}
    input_mask = 0
    analog: dict[str, dict[str, float]] = {"left": {}, "right": {}}
    sticks: dict[str, tuple[float, float]] = {}

    for field in fields:
        if field.number == BUTTON_MASK_FIELD:
            if field.wire_type != 0:
                raise WireDecodeError("field 6 button mask must be a varint")
            input_mask |= int(field.value)
            continue

        if field.number in POSE_FIELDS:
            poses_by_field.setdefault(field.number, _decode_pose_field(field))
            continue

        stick_side = STICK_FIELDS.get(field.number)
        if stick_side is not None:
            if field.wire_type != 2 or len(field.raw) != 8:
                raise WireDecodeError(f"field {field.number} stick must contain two float32 values")
            values = np.asarray(np.frombuffer(field.raw, dtype="<f4"), dtype=np.float64)
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise WireDecodeError(f"field {field.number} stick contains non-finite values")
            sticks[stick_side] = (float(values[0]), float(values[1]))
            continue

        for side, field_map in ANALOG_FIELD_BY_SIDE.items():
            for control, number in field_map.items():
                if field.number != number:
                    continue
                if field.wire_type != 5 or not math.isfinite(float(field.value)):
                    raise WireDecodeError(
                        f"analog field {field.number} ({side} {control}) is invalid"
                    )
                analog[side][control] = float(np.clip(float(field.value), 0.0, 1.0))

    poses: dict[str, PoseValue] = {}
    if 14 in poses_by_field:
        poses["head"] = poses_by_field[14]
    for side, candidates in CONTROLLER_POSE_CANDIDATES.items():
        for number in candidates:
            if number in poses_by_field:
                poses[side] = poses_by_field[number]
                break
        if side not in poses:
            raise WireDecodeError(f"missing {side} controller pose fields {candidates}")

    return DecodedSecDevPose(
        poses=poses,
        input_mask=input_mask,
        analog=analog,
        sticks=sticks,
    )


def resolve_controller_analog(
    packet: DecodedSecDevPose,
    side: str,
    control: str,
    previous: float,
    *,
    released_value: float = 0.0,
) -> float:
    """Resolve optional analog fields without latching a released control.

    Non-zero trigger/squeeze values are normally carried by float fields 10-13.
    When a held value is omitted, its field-6 bit distinguishes that case from a
    release. A release explicitly returns ``released_value`` instead of keeping
    the last closing command.
    """
    try:
        pressed_mask = ANALOG_MASK_BY_SIDE[side][control]
        value = packet.analog[side].get(control)
    except KeyError as exc:
        raise ValueError(f"unsupported controller analog: {side} {control}") from exc

    if value is not None:
        return float(np.clip(value, 0.0, 1.0))
    if packet.input_mask & pressed_mask:
        return float(np.clip(previous, 0.0, 1.0))
    return float(np.clip(released_value, 0.0, 1.0))
