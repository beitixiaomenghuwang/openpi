"""Minimal protobuf-wire decoder shared by S100 protocol consumers."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any


class WireDecodeError(ValueError):
    """Raised when a payload is not a valid protobuf-like wire message."""


@dataclass(frozen=True)
class WireField:
    number: int
    wire_type: int
    raw: bytes
    value: Any


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    start = offset
    while offset < len(data) and shift <= 63:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise WireDecodeError(f"unterminated varint at byte {start}")


def parse_wire_fields(data: bytes) -> list[WireField]:
    """Parse top-level protobuf wire fields without requiring a .proto file."""
    fields: list[WireField] = []
    offset = 0
    while offset < len(data):
        key, key_end = _read_varint(data, offset)
        number = key >> 3
        wire_type = key & 0x07
        if number <= 0:
            raise WireDecodeError(f"invalid field number {number} at byte {offset}")
        offset = key_end

        if wire_type == 0:
            value, value_end = _read_varint(data, offset)
            fields.append(WireField(number, wire_type, data[offset:value_end], value))
            offset = value_end
        elif wire_type == 1:
            end = offset + 8
            if end > len(data):
                raise WireDecodeError(f"truncated fixed64 field {number}")
            raw = data[offset:end]
            fields.append(WireField(number, wire_type, raw, struct.unpack("<d", raw)[0]))
            offset = end
        elif wire_type == 2:
            length, value_start = _read_varint(data, offset)
            end = value_start + length
            if end > len(data):
                raise WireDecodeError(f"truncated bytes field {number}")
            raw = data[value_start:end]
            fields.append(WireField(number, wire_type, raw, raw))
            offset = end
        elif wire_type == 5:
            end = offset + 4
            if end > len(data):
                raise WireDecodeError(f"truncated fixed32 field {number}")
            raw = data[offset:end]
            fields.append(WireField(number, wire_type, raw, struct.unpack("<f", raw)[0]))
            offset = end
        else:
            raise WireDecodeError(
                f"unsupported wire type {wire_type} for field {number}"
            )
    return fields
