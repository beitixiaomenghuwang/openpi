"""Parser for the TeleAvatar WebXR ``/devicepose`` binary protocol."""

from __future__ import annotations

import struct
from enum import IntEnum, IntFlag


class PackageType(IntEnum):
    XR_DEVICEPOSE = 13
    XR_HAPTIC_ACTUATORS = 14


class InputFlag(IntFlag):
    ENABLE = 0x0001
    HAND = 0x0002
    GRASP_SPACE = 0x0004
    AIM_SPACE = 0x0008


class Button(IntFlag):
    A_CLICK = 0x00000001
    A_TOUCH = 0x00000002
    B_CLICK = 0x00000004
    B_TOUCH = 0x00000008
    X_CLICK = 0x00000010
    X_TOUCH = 0x00000020
    Y_CLICK = 0x00000040
    Y_TOUCH = 0x00000080
    TRIGGER_CLICK = 0x00000100
    TRIGGER_TOUCH = 0x00000200
    SQUEEZE_CLICK = 0x00000400
    SQUEEZE_TOUCH = 0x00000800
    THUMB_CLICK = 0x00001000
    THUMB_TOUCH = 0x00002000
    SYS_CLICK = 0x00004000
    MENU_CLICK = 0x00008000
    PINCH = 0x00010000


def _vector3(data: bytes, offset: int) -> tuple[float, float, float]:
    return struct.unpack_from("<fff", data, offset)


def _quaternion(data: bytes, offset: int) -> tuple[float, float, float, float]:
    return struct.unpack_from("<ffff", data, offset)


def _pose(data: bytes, offset: int) -> dict:
    return {
        "quaternion": _quaternion(data, offset),
        "position": _vector3(data, offset + 16),
    }


def _controller(data: bytes, offset: int) -> dict:
    flags = struct.unpack_from("<I", data, offset)[0]
    buttons = struct.unpack_from("<Q", data, offset + 4)[0]
    stick_x, stick_y = struct.unpack_from("<ff", data, offset + 12)
    trigger = struct.unpack_from("<f", data, offset + 20)[0]
    squeeze = struct.unpack_from("<f", data, offset + 24)[0]
    return {
        "flags": flags,
        "buttons": buttons,
        "stick": (stick_x, stick_y),
        "trigger": trigger,
        "squeeze": squeeze,
        "squeeze_pose": _pose(data, offset + 28),
        "aim_pose": _pose(data, offset + 56),
    }


def parse_device_pose(data: bytes) -> dict:
    if len(data) < 208:
        raise ValueError(f"XR device pose packet is too short: {len(data)} < 208")
    message_type = struct.unpack_from("<I", data, 0)[0]
    if message_type != PackageType.XR_DEVICEPOSE:
        raise ValueError(f"Unexpected XR packet type: {message_type}")
    return {
        "hmd": _pose(data, 4),
        "left": _controller(data, 32),
        "right": _controller(data, 116),
        "hmd_type": struct.unpack_from("<I", data, 200)[0],
        "device_index": struct.unpack_from("<I", data, 204)[0],
    }


def make_haptic(controller_index: int, duration_ms: int = 100, strength: float = 0.5) -> bytes:
    value = max(0, min(100, round(float(strength) * 100.0)))
    return struct.pack(
        "<IIIII",
        int(PackageType.XR_HAPTIC_ACTUATORS),
        int(controller_index),
        0,
        value,
        int(duration_ms),
    )
