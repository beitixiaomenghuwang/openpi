"""Dependency-free SE(3) interpolation for command-side EE waypoints.

The policy emits absolute EE waypoints at a relatively low rate while the ROS
interface publishes at a higher rate. ``SE3Interpolator`` is the command-side
state between those two clocks: it traverses each target geometrically over one
control period, with linear position/trigger and shortest-path slerp for
orientation via quaternion exp/log maps. It has no velocity, acceleration or
jerk state, so it adds no tracking lag and imposes no rate limits of its own.
The implementation intentionally has no ROS dependencies so it can be tested
offline and keeps the interpolation callback lightweight.
"""

from __future__ import annotations

import numpy as np


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"Expected a finite quaternion with shape (4,), got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("Quaternion has near-zero norm")
    return quaternion / norm


def _quaternion_conjugate(value: np.ndarray) -> np.ndarray:
    quaternion = _normalize_quaternion(value)
    return np.array((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]), dtype=np.float64)


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = _normalize_quaternion(first)
    x2, y2, z2, w2 = _normalize_quaternion(second)
    return _normalize_quaternion(
        np.array(
            (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ),
            dtype=np.float64,
        )
    )


def _rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    value = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(value))
    if angle < 1e-8:
        return _normalize_quaternion(np.array((0.5 * value[0], 0.5 * value[1], 0.5 * value[2], 1.0)))
    half_angle = 0.5 * angle
    return np.concatenate((value / angle * np.sin(half_angle), np.array((np.cos(half_angle),))))


def _quaternion_to_rotvec(value: np.ndarray) -> np.ndarray:
    """Return the shortest-path rotation vector represented by ``value``."""
    quaternion = _normalize_quaternion(value)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    sine_half = float(np.linalg.norm(quaternion[:3]))
    if sine_half < 1e-8:
        return 2.0 * quaternion[:3]
    angle = 2.0 * float(np.arctan2(sine_half, quaternion[3]))
    return quaternion[:3] * (angle / sine_half)


def _validate_bimanual_action(action: np.ndarray) -> np.ndarray:
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (16,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Expected a finite 16D quaternion action, got shape {value.shape}")
    for start in (0, 8):
        if float(np.linalg.norm(value[start + 3 : start + 7])) < 1e-8:
            raise ValueError("Quaternion action contains a near-zero quaternion")
    return value


class SE3Interpolator:
    """Stateful bimanual target generator for ``left(8D) + right(8D)`` actions.

    A target is held until the next model tick. Each target is traversed over
    exactly one ``duration`` (the control period), so the command reaches the
    model waypoint by the time the next one arrives and adds no tracking lag.
    Position and trigger are linear, orientation is shortest-path slerp.

    The published stream is deliberately unbounded: a large model jump is passed
    on at whatever rate it implies. Command-side safety is the caller's job (see
    ``ee_quaternion_target_errors`` in the ROS interface).
    """

    def __init__(self, *, duration: float) -> None:
        if float(duration) <= 0.0 or not np.isfinite(float(duration)):
            raise ValueError("Interpolation duration must be positive and finite")
        self.duration = float(duration)
        self._start: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._command: np.ndarray | None = None
        self._elapsed = 0.0

    @property
    def initialized(self) -> bool:
        return self._command is not None

    def reset(self, action: np.ndarray | None = None) -> None:
        if action is None:
            self._start = None
            self._target = None
            self._command = None
            self._elapsed = 0.0
            return
        value = _validate_bimanual_action(action)
        for start in (0, 8):
            value[start + 3 : start + 7] = _normalize_quaternion(value[start + 3 : start + 7])
            value[start + 7] = np.clip(value[start + 7], 0.0, 1.0)
        self._start = value.copy()
        self._target = value.copy()
        self._command = value.copy()
        self._elapsed = 0.0

    def set_target(self, action: np.ndarray) -> None:
        value = _validate_bimanual_action(action)
        if self._command is None:
            self.reset(value)
            return
        for start in (0, 8):
            quaternion = _normalize_quaternion(value[start + 3 : start + 7])
            # Interpolate along the shortest arc from the command actually published.
            if float(np.dot(self._command[start + 3 : start + 7], quaternion)) < 0.0:
                quaternion = -quaternion
            value[start + 3 : start + 7] = quaternion
            value[start + 7] = np.clip(value[start + 7], 0.0, 1.0)
        # Restart the traversal from where the previous one stopped, so a target
        # that arrives early or late still produces a continuous command stream.
        self._start = self._command.copy()
        self._target = value
        self._elapsed = 0.0

    def step(self, dt: float) -> np.ndarray:
        if self._start is None or self._target is None or self._command is None:
            raise RuntimeError("Interpolator must be initialized before stepping")
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Interpolation step dt must be finite and positive")
        self._elapsed += dt
        alpha = min(self._elapsed / self.duration, 1.0)

        command = np.empty(16, dtype=np.float64)
        for start in (0, 8):
            start_quaternion = self._start[start + 3 : start + 7]
            command[start : start + 3] = (1.0 - alpha) * self._start[start : start + 3] + alpha * self._target[
                start : start + 3
            ]
            rotation = _quaternion_to_rotvec(
                _quaternion_multiply(self._target[start + 3 : start + 7], _quaternion_conjugate(start_quaternion))
            )
            command[start + 3 : start + 7] = _quaternion_multiply(
                _rotvec_to_quaternion(alpha * rotation), start_quaternion
            )
            command[start + 7] = np.clip(
                (1.0 - alpha) * self._start[start + 7] + alpha * self._target[start + 7], 0.0, 1.0
            )
        self._command = command
        return command.copy()
