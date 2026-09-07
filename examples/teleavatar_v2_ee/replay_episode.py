#!/usr/bin/env python3
"""
Inspect or replay one recorded TeleAvatar V2 end-effector episode.

The LeRobot converter keeps the raw 62D/72D TeleAvatar layout. This tool
extracts both absolute end-effector poses from indices 48:55 and 55:62 and
publishes them through the direct EE API topics:

- ``/api/left_arm/target_pose`` and ``/api/right_arm/target_pose``
- ``/api/left_gripper/cmd`` and ``/api/right_gripper/cmd``
- ``/api/fsm/enable``

For ``--source action``, gripper effort at indices 39 and 47 is converted to
the platform's normalized trigger. For ``--source state``, measured openness at
indices 7 and 15 is inverted to the same trigger convention.

The explicit ``--dry-run`` mode only reads and summarizes the selected episode;
it does not import ROS2 or publish commands. For real replay, the script first
ramp-interpolates both current EE poses to the first selected frame.

Examples:
    python examples/teleavatar_v2_ee/replay_episode.py \
        --dataset <dataset_path> --episode 0 --dry-run
    python examples/teleavatar_v2_ee/replay_episode.py \
        --dataset <dataset_path> --episode 0 --source action
    python examples/teleavatar_v2_ee/replay_episode.py \
        --dataset <dataset_path> --episode 12 --speed 0.5 --start 100

The replay path requires pandas + pyarrow for parquet reading and a running
ROS2/zenoh connection to the robot. Keep ``--dry-run`` enabled while checking
that the dataset poses and gripper ranges are sensible.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

EE_POSE_SLICES = {
    "left": slice(48, 55),
    "right": slice(55, 62),
}
GRIPPER_POSITION_INDEX = {"left": 7, "right": 15}
GRIPPER_EFFORT_INDEX = {"left": 39, "right": 47}


def gripper_effort_to_trigger(effort: np.ndarray) -> np.ndarray:
    """Convert recorded V2 gripper effort (Nm) to an API trigger in [0, 1]."""
    effort = np.asarray(effort, dtype=np.float32)
    trigger = np.where(effort > 0, 0.10 * (1.0 - effort / 2.0), 0.10 - effort * 0.90 / 1.6)
    return np.clip(trigger, 0.0, 1.0)


def gripper_openness_to_trigger(openness: np.ndarray) -> np.ndarray:
    """Convert measured openness (1=open, 0=closed) to the API trigger."""
    return 1.0 - np.clip(np.asarray(openness, dtype=np.float32), 0.0, 1.0)


def _normalize_quaternions(quaternions: np.ndarray, *, name: str) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float32)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if not np.all(np.isfinite(quaternions)) or np.any(norms < 1e-6):
        raise ValueError(f"{name} contains non-finite or zero-length quaternions")
    return quaternions / norms


def load_episode(
    dataset_root: pathlib.Path, episode: int, source: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load one episode's EE trajectory.

    Returns ``(left_pos, left_quat, right_pos, right_quat, left_trigger,
    right_trigger, fps)``. Positions are ``[T, 3]``, quaternions are ROS
    ``xyzw`` ``[T, 4]``, and triggers are ``[T]``.
    """
    if source not in {"action", "state"}:
        raise ValueError(f"Unsupported source {source!r}; use action or state")
    if episode < 0:
        raise ValueError("episode must be non-negative")

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Dataset metadata contains invalid fps={fps!r}")
    chunks_size = int(info["chunks_size"])
    chunk = episode // chunks_size
    parquet_path = dataset_root / info["data_path"].format(
        episode_chunk=chunk,
        episode_index=episode,
    )
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode file not found: {parquet_path}")

    column = "action" if source == "action" else "observation.state"
    # Keep parquet dependencies out of --help and module import paths.
    import pandas as pd  # noqa: PLC0415

    frame_values = pd.read_parquet(parquet_path, columns=[column])[column].to_numpy()
    if len(frame_values) == 0:
        raise ValueError(f"Episode {episode} contains no frames")
    data = np.stack(frame_values).astype(np.float32)
    if data.ndim != 2 or data.shape[1] < 62:
        raise ValueError(f"Expected {column} with shape [T, >=62], got {data.shape}")

    left_pose = data[:, EE_POSE_SLICES["left"]]
    right_pose = data[:, EE_POSE_SLICES["right"]]
    left_pos, left_quat = left_pose[:, :3], _normalize_quaternions(left_pose[:, 3:], name="left EE pose")
    right_pos, right_quat = right_pose[:, :3], _normalize_quaternions(right_pose[:, 3:], name="right EE pose")
    if not np.all(np.isfinite(left_pos)) or not np.all(np.isfinite(right_pos)):
        raise ValueError("EE positions contain non-finite values")

    if source == "action":
        left_trigger = gripper_effort_to_trigger(data[:, GRIPPER_EFFORT_INDEX["left"]])
        right_trigger = gripper_effort_to_trigger(data[:, GRIPPER_EFFORT_INDEX["right"]])
    else:
        left_trigger = gripper_openness_to_trigger(data[:, GRIPPER_POSITION_INDEX["left"]])
        right_trigger = gripper_openness_to_trigger(data[:, GRIPPER_POSITION_INDEX["right"]])

    return left_pos, left_quat, right_pos, right_quat, left_trigger, right_trigger, fps


def _quaternion_nlerp(start: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-path normalized quaternion interpolation for a safe ramp."""
    start = _normalize_quaternions(start, name="ramp start quaternion")
    target = _normalize_quaternions(target, name="ramp target quaternion")
    if float(np.dot(start, target)) < 0.0:
        target = -target
    return _normalize_quaternions((1.0 - alpha) * start + alpha * target, name="ramp quaternion")


def print_summary(
    left_pos: np.ndarray,
    left_quat: np.ndarray,
    right_pos: np.ndarray,
    right_quat: np.ndarray,
    left_trigger: np.ndarray,
    right_trigger: np.ndarray,
    fps: float,
    speed: float,
) -> None:
    """Print dimensions and motion ranges useful for dataset sanity checks."""
    frames = len(left_pos)
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    duration = frames / (fps * speed)
    positions = np.concatenate((left_pos, right_pos), axis=1)
    position_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1) if frames > 1 else np.zeros(1)
    print(f"frames={frames}  fps={fps:g}  speed={speed:g}x  duration={duration:.1f}s")
    print(f"left  position first={np.round(left_pos[0], 4).tolist()}")
    print(f"left  position last ={np.round(left_pos[-1], 4).tolist()}")
    print(f"right position first={np.round(right_pos[0], 4).tolist()}")
    print(f"right position last ={np.round(right_pos[-1], 4).tolist()}")
    print(f"left  quaternion first={np.round(left_quat[0], 4).tolist()}")
    print(f"right quaternion first={np.round(right_quat[0], 4).tolist()}")
    print(f"left  trigger range=[{left_trigger.min():.3f}, {left_trigger.max():.3f}]")
    print(f"right trigger range=[{right_trigger.min():.3f}, {right_trigger.max():.3f}]")
    print(f"max combined per-frame EE translation step={position_steps.max() * 1000.0:.2f} mm")


def _build_pose_message(position: np.ndarray, quaternion: np.ndarray):
    from geometry_msgs.msg import Pose  # noqa: PLC0415

    message = Pose()
    message.position.x, message.position.y, message.position.z = map(float, position)
    message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w = map(
        float, quaternion
    )
    return message


def _replay_ros(
    left_pos: np.ndarray,
    left_quat: np.ndarray,
    right_pos: np.ndarray,
    right_quat: np.ndarray,
    left_trigger: np.ndarray,
    right_trigger: np.ndarray,
    fps: float,
    speed: float,
    ramp_s: float,
) -> None:
    """Ramp and replay an absolute bimanual EE trajectory over ROS2."""
    from geometry_msgs.msg import Pose  # noqa: PLC0415
    import rclpy  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from std_msgs.msg import Float32  # noqa: PLC0415

    rclpy.init()
    node = Node("teleavatar_ee_episode_replayer")
    publishers = {
        "left_pose": node.create_publisher(Pose, "/api/left_arm/target_pose", 10),
        "right_pose": node.create_publisher(Pose, "/api/right_arm/target_pose", 10),
        "left_gripper": node.create_publisher(Float32, "/api/left_gripper/cmd", 10),
        "right_gripper": node.create_publisher(Float32, "/api/right_gripper/cmd", 10),
        "enable": node.create_publisher(Float32, "/api/fsm/enable", 10),
    }
    current: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def on_pose(message, arm: str) -> None:
        position = np.array([message.position.x, message.position.y, message.position.z], dtype=np.float32)
        quaternion = np.array(
            [message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w],
            dtype=np.float32,
        )
        try:
            current[arm] = (position, _normalize_quaternions(quaternion, name=f"current {arm} pose"))
        except ValueError:
            node.get_logger().warning(f"Ignoring invalid current {arm} EE pose")

    node.create_subscription(
        Pose,
        "/left_arm/current_ee_pose",
        lambda message: on_pose(message, "left"),
        10,
    )
    node.create_subscription(
        Pose,
        "/right_arm/current_ee_pose",
        lambda message: on_pose(message, "right"),
        10,
    )

    def publish_frame(lp, lq, rp, rq, lt, rt) -> None:
        publishers["enable"].publish(Float32(data=1.0))
        publishers["left_pose"].publish(_build_pose_message(lp, lq))
        publishers["right_pose"].publish(_build_pose_message(rp, rq))
        publishers["left_gripper"].publish(Float32(data=float(lt)))
        publishers["right_gripper"].publish(Float32(data=float(rt)))

    try:
        node.get_logger().info("Waiting for /left_arm/current_ee_pose and /right_arm/current_ee_pose...")
        deadline = time.monotonic() + 10.0
        while len(current) < 2:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out waiting for current EE pose topics")

        if ramp_s < 0.0:
            raise ValueError("ramp_s must be non-negative")
        if ramp_s > 0.0:
            node.get_logger().info(f"Ramping both EE poses to frame 0 over {ramp_s:.1f}s...")
            start_left, start_right = current["left"], current["right"]
            ramp_start = time.monotonic()
            while (elapsed := time.monotonic() - ramp_start) < ramp_s:
                alpha = min(elapsed / ramp_s, 1.0)
                left_ramp_q = _quaternion_nlerp(start_left[1], left_quat[0], alpha)
                right_ramp_q = _quaternion_nlerp(start_right[1], right_quat[0], alpha)
                publish_frame(
                    start_left[0] * (1.0 - alpha) + left_pos[0] * alpha,
                    left_ramp_q,
                    start_right[0] * (1.0 - alpha) + right_pos[0] * alpha,
                    right_ramp_q,
                    left_trigger[0],
                    right_trigger[0],
                )
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(0.02)

        period = 1.0 / (fps * speed)
        node.get_logger().info(f"Replaying {len(left_pos)} EE frames at {fps * speed:g} Hz...")
        replay_start = time.monotonic()
        for index in range(len(left_pos)):
            publish_frame(
                left_pos[index],
                left_quat[index],
                right_pos[index],
                right_quat[index],
                left_trigger[index],
                right_trigger[index],
            )
            if index % 100 == 0:
                node.get_logger().info(f"  frame {index}/{len(left_pos)}")
            sleep = replay_start + (index + 1) * period - time.monotonic()
            if sleep > 0.0:
                time.sleep(sleep)
        node.get_logger().info("Replay finished; robot holds the last commanded EE pose.")
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted; stopped publishing. Robot holds the last commanded EE pose.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or replay a TeleAvatar V2 EE episode")
    parser.add_argument("--dataset", type=pathlib.Path, required=True, help="LeRobot dataset root")
    parser.add_argument("--episode", type=int, required=True, help="Episode index")
    parser.add_argument(
        "--source",
        choices=["action", "state"],
        default="action",
        help="Read absolute EE targets from action or measured observation.state",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed factor")
    parser.add_argument(
        "--ramp-s",
        type=float,
        default=5.0,
        help="Seconds to ramp from current EE poses to frame 0 (0 disables ramp)",
    )
    parser.add_argument("--start", type=int, default=0, help="First frame to replay")
    parser.add_argument("--end", type=int, default=None, help="Exclusive end frame")
    parser.add_argument("--yes", action="store_true", help="Skip the replay confirmation prompt")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the selected episode summary; publish nothing",
    )
    args = parser.parse_args()

    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.end is not None and args.end <= args.start:
        parser.error("--end must be greater than --start")

    trajectory = load_episode(args.dataset, args.episode, args.source)
    left_pos, left_quat, right_pos, right_quat, left_trigger, right_trigger, fps = trajectory
    selected = slice(args.start, args.end)
    left_pos, left_quat = left_pos[selected], left_quat[selected]
    right_pos, right_quat = right_pos[selected], right_quat[selected]
    left_trigger, right_trigger = left_trigger[selected], right_trigger[selected]
    if len(left_pos) == 0:
        raise ValueError("Selected frame range is empty")

    end_text = "" if args.end is None else str(args.end)
    print(f"Episode {args.episode} ({args.source}) from {args.dataset}, frames [{args.start}:{end_text}]")
    print_summary(left_pos, left_quat, right_pos, right_quat, left_trigger, right_trigger, fps, args.speed)
    if args.dry_run:
        return

    if not args.yes:
        input("Robot will ramp both EE poses and replay. Press Enter to start (Ctrl+C to abort)... ")

    _replay_ros(
        left_pos,
        left_quat,
        right_pos,
        right_quat,
        left_trigger,
        right_trigger,
        fps,
        args.speed,
        args.ramp_s,
    )


if __name__ == "__main__":
    main()
