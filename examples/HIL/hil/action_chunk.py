"""Validated JSON contract shared by policy, teleop, and the HIL supervisor."""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    from .pose_math import as_vector, normalize_quaternion
except ImportError:
    from pose_math import as_vector, normalize_quaternion


SCHEMA = "teleavatar.hil.ee_action_chunk.v1"
SIDES = ("left", "right")
PROCESS_SESSION_ID = uuid.uuid4().hex


@dataclass(frozen=True)
class EndpointCommand:
    position: np.ndarray
    quaternion: np.ndarray
    gripper: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", as_vector(self.position, 3).copy())
        object.__setattr__(
            self, "quaternion", normalize_quaternion(self.quaternion).copy()
        )
        value = float(self.gripper)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("gripper must be finite and in [0, 1]")
        object.__setattr__(self, "gripper", value)


@dataclass(frozen=True)
class ActionFrame:
    left: EndpointCommand
    right: EndpointCommand

    def command(self, side: str) -> EndpointCommand:
        if side not in SIDES:
            raise KeyError(side)
        return self.left if side == "left" else self.right


@dataclass(frozen=True)
class ActionChunk:
    source: str
    session_id: str
    sequence: int
    control_hz: float
    frame_ids: Mapping[str, str]
    actions: tuple[ActionFrame, ...]
    created_at_ns: int


def _read_endpoint(raw: Mapping[str, object]) -> EndpointCommand:
    if not isinstance(raw, Mapping):
        raise ValueError("endpoint command must be an object")
    return EndpointCommand(
        position=raw.get("position", ()),
        quaternion=raw.get("quaternion", ()),
        gripper=raw.get("gripper", math.nan),
    )


def decode_action_chunk(
    payload: str | bytes,
    *,
    expected_source: str | None = None,
    expected_frame_ids: Mapping[str, str] | None = None,
    max_actions: int = 240,
) -> ActionChunk:
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    try:
        raw = json.loads(payload)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid action chunk JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("action chunk root must be an object")
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"unsupported action chunk schema: {raw.get('schema')!r}")

    source = raw.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("action chunk source must be a non-empty string")
    if expected_source is not None and source != expected_source:
        raise ValueError(f"action chunk source {source!r} != {expected_source!r}")

    session_id = raw.get("session_id")
    if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
        raise ValueError("action chunk session_id must contain 1 to 128 characters")

    sequence = raw.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("action chunk sequence must be a non-negative integer")
    control_hz = float(raw.get("control_hz", math.nan))
    if not math.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("action chunk control_hz must be positive")
    created_at_ns = raw.get("created_at_ns", 0)
    if (
        isinstance(created_at_ns, bool)
        or not isinstance(created_at_ns, int)
        or created_at_ns < 0
    ):
        raise ValueError("action chunk created_at_ns must be a non-negative integer")
    frame_ids = raw.get("frame_ids")
    if not isinstance(frame_ids, Mapping):
        raise ValueError("action chunk frame_ids must be an object")
    parsed_frames = {}
    for side in SIDES:
        frame_id = frame_ids.get(side)
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError(f"missing frame id for {side}")
        parsed_frames[side] = frame_id
    if expected_frame_ids is not None:
        for side in SIDES:
            expected = expected_frame_ids[side]
            if parsed_frames[side] != expected:
                raise ValueError(
                    f"{side} frame {parsed_frames[side]!r} != expected {expected!r}"
                )

    raw_actions = raw.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("action chunk actions must be a non-empty list")
    if len(raw_actions) > max_actions:
        raise ValueError(
            f"action chunk contains {len(raw_actions)} actions; limit is {max_actions}"
        )
    actions = []
    for index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, Mapping):
            raise ValueError(f"action {index} must be an object")
        try:
            actions.append(
                ActionFrame(
                    left=_read_endpoint(raw_action.get("left", {})),
                    right=_read_endpoint(raw_action.get("right", {})),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid action {index}: {exc}") from exc

    return ActionChunk(
        source=source,
        session_id=session_id,
        sequence=sequence,
        control_hz=control_hz,
        frame_ids=parsed_frames,
        actions=tuple(actions),
        created_at_ns=created_at_ns,
    )


def encode_action_chunk(
    *,
    source: str,
    session_id: str | None = None,
    sequence: int,
    control_hz: float,
    frame_ids: Mapping[str, str],
    actions: Sequence[ActionFrame],
    created_at_ns: int | None = None,
) -> str:
    chunk = ActionChunk(
        source=str(source),
        session_id=PROCESS_SESSION_ID if session_id is None else str(session_id),
        sequence=int(sequence),
        control_hz=float(control_hz),
        frame_ids=dict(frame_ids),
        actions=tuple(actions),
        created_at_ns=time.time_ns() if created_at_ns is None else int(created_at_ns),
    )
    if (
        not chunk.source
        or not chunk.session_id
        or len(chunk.session_id) > 128
        or chunk.sequence < 0
        or not math.isfinite(chunk.control_hz)
        or chunk.control_hz <= 0.0
    ):
        raise ValueError("invalid action chunk metadata")
    if not chunk.actions:
        raise ValueError("action chunk must contain at least one action")
    for side in SIDES:
        if not isinstance(chunk.frame_ids.get(side), str) or not chunk.frame_ids[side]:
            raise ValueError(f"missing frame id for {side}")

    raw = {
        "schema": SCHEMA,
        "source": chunk.source,
        "session_id": chunk.session_id,
        "sequence": chunk.sequence,
        "control_hz": chunk.control_hz,
        "created_at_ns": chunk.created_at_ns,
        "frame_ids": dict(chunk.frame_ids),
        "actions": [],
    }
    for action in chunk.actions:
        encoded_action = {}
        for side in SIDES:
            command = action.command(side)
            encoded_action[side] = {
                "position": command.position.tolist(),
                "quaternion": command.quaternion.tolist(),
                "gripper": command.gripper,
            }
        raw["actions"].append(encoded_action)
    return json.dumps(raw, separators=(",", ":"), sort_keys=True, allow_nan=False)


def make_action_frame(
    *,
    left_position: Iterable[float],
    left_quaternion: Iterable[float],
    left_gripper: float,
    right_position: Iterable[float],
    right_quaternion: Iterable[float],
    right_gripper: float,
) -> ActionFrame:
    return ActionFrame(
        left=EndpointCommand(left_position, left_quaternion, left_gripper),
        right=EndpointCommand(right_position, right_quaternion, right_gripper),
    )
