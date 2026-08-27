#!/usr/bin/env python3
"""Combine the legacy four-topic policy output into one HIL action chunk.

The policy-facing topics are deliberately below ``/hil``. This node never
publishes robot ``/api`` topics; only ``hil_supervisor`` owns those outputs.
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from dataclasses import dataclass

try:
    import rclpy
    from geometry_msgs.msg import Pose
    from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, String
except ImportError:  # pragma: no cover - allows pure module inspection
    rclpy = None
    _rclpy = None
    Pose = Node = Float32 = String = None

try:
    from . import topics
    from .action_chunk import ActionFrame, EndpointCommand, encode_action_chunk
except ImportError:  # pragma: no cover
    import topics
    from action_chunk import ActionFrame, EndpointCommand, encode_action_chunk


_NodeBase = Node if Node is not None else object


@dataclass
class _Sample:
    value: object = None
    received_at: float = 0.0
    generation: int = 0


class FourTopicBuffer:
    """Return a candidate only after every input has produced a fresh value."""

    NAMES = ("left_pose", "right_pose", "left_gripper", "right_gripper")

    def __init__(self, sync_slop: float):
        if sync_slop <= 0.0:
            raise ValueError("sync_slop must be positive")
        self.sync_slop = float(sync_slop)
        self.lock = threading.RLock()
        self.samples = {name: _Sample() for name in self.NAMES}
        self.last_emitted_generation = {name: 0 for name in self.NAMES}

    def update(self, name: str, value: object, received_at: float | None = None) -> None:
        if name not in self.samples:
            raise KeyError(name)
        with self.lock:
            sample = self.samples[name]
            sample.value = value
            sample.received_at = (
                time.monotonic() if received_at is None else float(received_at)
            )
            sample.generation += 1

    def take(self) -> tuple[dict[str, object] | None, float | None]:
        """Return ``(values, spread)``; a non-None spread without values is a drop."""

        with self.lock:
            if any(sample.value is None for sample in self.samples.values()):
                return None, None
            generations = {
                name: sample.generation for name, sample in self.samples.items()
            }
            if any(
                generations[name] <= self.last_emitted_generation[name]
                for name in self.samples
            ):
                return None, None

            timestamps = [sample.received_at for sample in self.samples.values()]
            spread = max(timestamps) - min(timestamps)
            self.last_emitted_generation = generations
            if spread > self.sync_slop:
                return None, spread
            return {name: sample.value for name, sample in self.samples.items()}, spread


class PolicyTopicAdapter(_NodeBase):
    """Synchronize four policy topics and publish each complete set immediately."""

    def __init__(self, *, control_hz: float, sync_slop: float):
        super().__init__("hil_policy_topic_adapter")
        if control_hz <= 0.0 or sync_slop <= 0.0:
            raise ValueError("control_hz and sync_slop must be positive")

        self.control_hz = float(control_hz)
        self.input_buffer = FourTopicBuffer(sync_slop)
        self.session_id = uuid.uuid4().hex
        self.sequence = 0
        self.rejected_sync = 0
        self._last_sync_warning = 0.0

        self.chunk_publisher = self.create_publisher(
            String, topics.POLICY_INPUT_ACTION_CHUNK, 10
        )
        self.create_subscription(
            Pose,
            topics.POLICY_INPUT_LEFT_POSE,
            lambda msg: self._on_pose("left_pose", msg),
            10,
        )
        self.create_subscription(
            Pose,
            topics.POLICY_INPUT_RIGHT_POSE,
            lambda msg: self._on_pose("right_pose", msg),
            10,
        )
        self.create_subscription(
            Float32,
            topics.POLICY_INPUT_LEFT_GRIPPER,
            lambda msg: self._on_gripper("left_gripper", msg),
            10,
        )
        self.create_subscription(
            Float32,
            topics.POLICY_INPUT_RIGHT_GRIPPER,
            lambda msg: self._on_gripper("right_gripper", msg),
            10,
        )
        self.get_logger().info(
            "Event-driven four-topic policy adapter ready: "
            f"sync_slop={self.input_buffer.sync_slop:.3f}s "
            f"control_hz={self.control_hz:.1f}"
        )

    def _update(self, name: str, value: object) -> None:
        self.input_buffer.update(name, value)
        # take() succeeds only after every input has produced a fresh value, so
        # the callback carrying the fourth value publishes without a poll delay.
        self._try_publish()

    def _on_pose(self, name: str, msg) -> None:
        value = (
            (float(msg.position.x), float(msg.position.y), float(msg.position.z)),
            (
                float(msg.orientation.x),
                float(msg.orientation.y),
                float(msg.orientation.z),
                float(msg.orientation.w),
            ),
        )
        self._update(name, value)

    def _on_gripper(self, name: str, msg) -> None:
        self._update(name, float(msg.data))

    def _take_candidate(self):
        values, spread = self.input_buffer.take()
        if values is None and spread is not None:
            self.rejected_sync += 1
            now = time.monotonic()
            if now - self._last_sync_warning >= 1.0:
                self.get_logger().warning(
                    "Dropped unsynchronized policy values: "
                    f"spread={spread:.3f}s > {self.input_buffer.sync_slop:.3f}s"
                )
                self._last_sync_warning = now
        return values

    def _try_publish(self) -> None:
        values = self._take_candidate()
        if values is None:
            return

        try:
            left_position, left_quaternion = values["left_pose"]
            right_position, right_quaternion = values["right_pose"]
            frame = ActionFrame(
                left=EndpointCommand(
                    left_position,
                    left_quaternion,
                    values["left_gripper"],
                ),
                right=EndpointCommand(
                    right_position,
                    right_quaternion,
                    values["right_gripper"],
                ),
            )
            payload = encode_action_chunk(
                source="policy",
                session_id=self.session_id,
                sequence=self.sequence,
                control_hz=self.control_hz,
                frame_ids=topics.ACTION_FRAME_IDS["policy"],
                actions=[frame],
            )
        except (TypeError, ValueError) as exc:
            self.get_logger().error(f"Rejected policy four-topic action: {exc}")
            return

        self.chunk_publisher.publish(String(data=payload))
        self.sequence += 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--sync-slop", type=float, default=0.02)
    return parser.parse_args(argv)


def main() -> None:
    if rclpy is None:  # pragma: no cover
        raise RuntimeError("ROS2 Python dependencies are unavailable")
    args = parse_args()
    rclpy.init()
    node = PolicyTopicAdapter(
        control_hz=args.control_hz,
        sync_slop=args.sync_slop,
    )
    try:
        rclpy.spin(node)
    except (
        KeyboardInterrupt,
        rclpy.executors.ExternalShutdownException,
        _rclpy.RCLError,
    ):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
