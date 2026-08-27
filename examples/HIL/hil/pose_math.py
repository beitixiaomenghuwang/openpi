"""Small, dependency-light pose math helpers using xyzw quaternions."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EPS = 1e-9


def as_vector(values: Iterable[float], size: int) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Expected {size} finite values, got {vector}")
    return vector


def normalize_quaternion(quaternion: Iterable[float]) -> np.ndarray:
    q = as_vector(quaternion, 4)
    norm = float(np.linalg.norm(q))
    if norm < EPS:
        raise ValueError("Quaternion must not be all zero")
    return q / norm


def quaternion_conjugate(quaternion: Iterable[float]) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quaternion_multiply(left: Iterable[float], right: Iterable[float]) -> np.ndarray:
    x1, y1, z1, w1 = normalize_quaternion(left)
    x2, y2, z2, w2 = normalize_quaternion(right)
    result = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )
    return normalize_quaternion(result)


def quaternion_to_matrix(quaternion: Iterable[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError("Rotation matrix must be finite and 3x3")

    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        )
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = math.sqrt(max(EPS, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
            q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
        elif axis == 1:
            s = math.sqrt(max(EPS, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
            q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
        else:
            s = math.sqrt(max(EPS, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
            q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])
    return normalize_quaternion(q)


def change_basis(position: Iterable[float], quaternion: Iterable[float], basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    basis = np.asarray(basis, dtype=np.float64)
    if basis.shape != (3, 3):
        raise ValueError("Basis must be 3x3")
    p = basis @ as_vector(position, 3)
    rotation = basis @ quaternion_to_matrix(quaternion) @ basis.T
    return p, matrix_to_quaternion(rotation)


def apply_tool_rotation(quaternion: Iterable[float], tool_rotation: np.ndarray) -> np.ndarray:
    return matrix_to_quaternion(quaternion_to_matrix(quaternion) @ np.asarray(tool_rotation, dtype=np.float64))


def quaternion_angle(left: Iterable[float], right: Iterable[float]) -> float:
    q0 = normalize_quaternion(left)
    q1 = normalize_quaternion(right)
    dot = float(np.clip(abs(np.dot(q0, q1)), -1.0, 1.0))
    return 2.0 * math.acos(dot)


def quaternion_slerp(left: Iterable[float], right: Iterable[float], amount: float) -> np.ndarray:
    q0 = normalize_quaternion(left)
    q1 = normalize_quaternion(right)
    t = float(np.clip(amount, 0.0, 1.0))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion(q0 + t * (q1 - q0))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return normalize_quaternion(
        math.sin((1.0 - t) * theta) / sin_theta * q0
        + math.sin(t * theta) / sin_theta * q1
    )


def anchored_pose(
    raw_position: Iterable[float],
    raw_quaternion: Iterable[float],
    raw_anchor_position: Iterable[float],
    raw_anchor_quaternion: Iterable[float],
    robot_anchor_position: Iterable[float],
    robot_anchor_quaternion: Iterable[float],
    position_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a raw hand pose relative to its takeover anchor onto the robot."""
    raw_position = as_vector(raw_position, 3)
    raw_anchor_position = as_vector(raw_anchor_position, 3)
    robot_anchor_position = as_vector(robot_anchor_position, 3)
    position = robot_anchor_position + float(position_scale) * (raw_position - raw_anchor_position)
    rotation_delta = quaternion_multiply(quaternion_conjugate(raw_anchor_quaternion), raw_quaternion)
    quaternion = quaternion_multiply(robot_anchor_quaternion, rotation_delta)
    return position, quaternion


def limit_pose_step(
    previous_position: Iterable[float],
    previous_quaternion: Iterable[float],
    target_position: Iterable[float],
    target_quaternion: Iterable[float],
    max_translation: float,
    max_rotation: float,
) -> tuple[np.ndarray, np.ndarray]:
    previous_position = as_vector(previous_position, 3)
    target_position = as_vector(target_position, 3)
    delta = target_position - previous_position
    distance = float(np.linalg.norm(delta))
    if distance > max_translation > 0.0:
        target_position = previous_position + delta * (max_translation / distance)

    angle = quaternion_angle(previous_quaternion, target_quaternion)
    if angle > max_rotation > 0.0:
        target_quaternion = quaternion_slerp(previous_quaternion, target_quaternion, max_rotation / angle)
    else:
        target_quaternion = normalize_quaternion(target_quaternion)
    return target_position, target_quaternion


def clamp_arm_reach(position: Iterable[float], max_reach: float) -> np.ndarray:
    position = as_vector(position, 3)
    distance = float(np.linalg.norm(position))
    if distance > max_reach > 0.0:
        return position * (max_reach / distance)
    return position
