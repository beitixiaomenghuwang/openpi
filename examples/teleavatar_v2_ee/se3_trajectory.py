"""Dependency-free stateful SE(3) trajectory generation for EE commands.

The policy emits absolute EE waypoints at a relatively low rate while the ROS
interface publishes at a higher rate. ``SE3Trajectory`` is the command-side
state between those two clocks. Translation and rotation use the same bounded
velocity/acceleration/jerk tracking structure; orientation errors and updates
are represented with shortest-path rotation vectors and quaternion exp/log
maps. The implementation intentionally has no ROS dependencies so it can be
tested offline and keeps the interpolation callback lightweight.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"Expected a finite quaternion with shape (4,), got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("Quaternion has near-zero norm")
    return quaternion / norm


def _clip_vector_norm(value: np.ndarray, maximum: float) -> np.ndarray:
    """Clip a vector's norm without changing its direction."""
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm < 1e-12:
        return vector.copy()
    return vector * (maximum / norm)


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


@dataclasses.dataclass
class _ArmState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    quaternion: np.ndarray
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray
    trigger: float


class SE3Trajectory:
    """Stateful bimanual target generator for ``left(8D) + right(8D)`` actions.

    A target is held until the next model tick. Each interpolation tick moves
    the command state toward that target with bounded speed, acceleration and
    jerk. A zero limit disables that particular bound. ``minimum_duration`` is
    the policy/control period and also determines the response time (four
    control periods, about 89 ms at the default 45 Hz), which keeps the
    tracker well damped while retaining responsive target following.
    """

    def __init__(
        self,
        *,
        minimum_duration: float,
        max_translation_speed: float,
        max_rotation_speed: float,
        max_translation_acceleration: float,
        max_rotation_acceleration: float,
        max_translation_jerk: float,
        max_rotation_jerk: float,
        max_trigger_speed: float = 5.0,
    ) -> None:
        limits = (
            max_translation_speed,
            max_rotation_speed,
            max_translation_acceleration,
            max_rotation_acceleration,
            max_translation_jerk,
            max_rotation_jerk,
            max_trigger_speed,
        )
        if float(minimum_duration) <= 0.0 or any(
            float(value) < 0.0 or not np.isfinite(float(value)) for value in limits
        ):
            raise ValueError("Trajectory duration must be positive and limits must be non-negative")
        self.minimum_duration = float(minimum_duration)
        self.response_time = max(4.0 * self.minimum_duration, 0.05)
        self.max_translation_speed = float(max_translation_speed)
        self.max_rotation_speed = float(max_rotation_speed)
        self.max_translation_acceleration = float(max_translation_acceleration)
        self.max_rotation_acceleration = float(max_rotation_acceleration)
        self.max_translation_jerk = float(max_translation_jerk)
        self.max_rotation_jerk = float(max_rotation_jerk)
        self.max_trigger_speed = float(max_trigger_speed)
        self._states: list[_ArmState] | None = None
        self._target: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self._states is not None

    @staticmethod
    def _validate_action(action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (16,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Expected a finite 16D quaternion action, got shape {value.shape}")
        for start in (0, 8):
            if float(np.linalg.norm(value[start + 3 : start + 7])) < 1e-8:
                raise ValueError("Quaternion action contains a near-zero quaternion")
        return value

    @staticmethod
    def _make_state(action: np.ndarray, start: int) -> _ArmState:
        return _ArmState(
            position=action[start : start + 3].copy(),
            velocity=np.zeros(3, dtype=np.float64),
            acceleration=np.zeros(3, dtype=np.float64),
            quaternion=_normalize_quaternion(action[start + 3 : start + 7]),
            angular_velocity=np.zeros(3, dtype=np.float64),
            angular_acceleration=np.zeros(3, dtype=np.float64),
            trigger=float(np.clip(action[start + 7], 0.0, 1.0)),
        )

    def reset(self, action: np.ndarray | None = None) -> None:
        if action is None:
            self._states = None
            self._target = None
            return
        value = self._validate_action(action)
        self._states = [self._make_state(value, 0), self._make_state(value, 8)]
        self._target = value.copy()

    @staticmethod
    def _desired_velocity(
        error: np.ndarray,
        response_time: float,
        max_speed: float,
        max_acceleration: float,
    ) -> np.ndarray:
        distance = float(np.linalg.norm(error))
        if distance < 1e-9:
            return np.zeros_like(error)
        desired_speed = distance / response_time
        if max_speed > 0.0 and np.isfinite(max_speed):
            desired_speed = min(desired_speed, max_speed)
        # Do not ask the state to move faster than it can stop under the
        # configured acceleration bound. This keeps a stationary target from
        # being chased with an unnecessarily high residual speed.
        if max_acceleration > 0.0 and np.isfinite(max_acceleration):
            desired_speed = min(desired_speed, float(np.sqrt(2.0 * max_acceleration * distance)))
        return error * (desired_speed / distance)

    @staticmethod
    def _update_derivative(
        value: np.ndarray,
        derivative: np.ndarray,
        desired_derivative: np.ndarray,
        dt: float,
        response_time: float,
        max_derivative: float,
        max_jerk: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        desired_acceleration = (desired_derivative - derivative) / response_time
        if max_derivative > 0.0 and np.isfinite(max_derivative):
            desired_acceleration = _clip_vector_norm(desired_acceleration, max_derivative)
        if max_jerk > 0.0 and np.isfinite(max_jerk):
            value = value + _clip_vector_norm(desired_acceleration - value, max_jerk * dt)
        else:
            value = desired_acceleration
        if max_derivative > 0.0 and np.isfinite(max_derivative):
            value = _clip_vector_norm(value, max_derivative)
        derivative = derivative + value * dt
        if max_derivative > 0.0 and np.isfinite(max_derivative):
            derivative = _clip_vector_norm(derivative, max_derivative)
        return value, derivative

    def set_target(self, action: np.ndarray) -> None:
        value = self._validate_action(action).copy()
        if self._states is None:
            self.reset(value)
            return
        for state, start in zip(self._states, (0, 8)):
            target_quaternion = _normalize_quaternion(value[start + 3 : start + 7])
            if float(np.dot(state.quaternion, target_quaternion)) < 0.0:
                target_quaternion = -target_quaternion
            value[start + 3 : start + 7] = target_quaternion
            value[start + 7] = np.clip(value[start + 7], 0.0, 1.0)
        self._target = value

    def step(self, dt: float) -> np.ndarray:
        if self._states is None or self._target is None:
            raise RuntimeError("Trajectory must be initialized before stepping")
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Trajectory step dt must be finite and positive")
        for state, start in zip(self._states, (0, 8)):
            position_error = self._target[start : start + 3] - state.position
            desired_velocity = self._desired_velocity(
                position_error,
                self.response_time,
                self.max_translation_speed,
                self.max_translation_acceleration,
            )
            state.acceleration, state.velocity = self._update_derivative(
                state.acceleration,
                state.velocity,
                desired_velocity,
                dt,
                self.response_time,
                self.max_translation_acceleration,
                self.max_translation_jerk,
            )
            # Do not snap to the target when the residual error is tiny: that
            # would create a finite-difference velocity/acceleration spike in
            # the published stream. The stopping-distance bound keeps the
            # residual speed small as a stationary target is approached.
            state.position = state.position + state.velocity * dt

            rotation_error = _quaternion_to_rotvec(
                _quaternion_multiply(self._target[start + 3 : start + 7], _quaternion_conjugate(state.quaternion))
            )
            desired_angular_velocity = self._desired_velocity(
                rotation_error,
                self.response_time,
                self.max_rotation_speed,
                self.max_rotation_acceleration,
            )
            state.angular_acceleration, state.angular_velocity = self._update_derivative(
                state.angular_acceleration,
                state.angular_velocity,
                desired_angular_velocity,
                dt,
                self.response_time,
                self.max_rotation_acceleration,
                self.max_rotation_jerk,
            )
            next_rotation = _rotvec_to_quaternion(state.angular_velocity * dt)
            state.quaternion = _quaternion_multiply(next_rotation, state.quaternion)

            trigger_delta = float(self._target[start + 7] - state.trigger)
            if self.max_trigger_speed > 0.0:
                trigger_delta = float(np.clip(trigger_delta, -self.max_trigger_speed * dt, self.max_trigger_speed * dt))
            state.trigger = float(np.clip(state.trigger + trigger_delta, 0.0, 1.0))

        command = np.empty(16, dtype=np.float64)
        for state, start in zip(self._states, (0, 8)):
            command[start : start + 3] = state.position
            command[start + 3 : start + 7] = state.quaternion
            command[start + 7] = state.trigger
        return command
