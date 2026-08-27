#!/usr/bin/env python3
"""Receive second-S100 VR poses and publish teleop EE action chunks.

The second S100 sends its native stereo video directly to the headset. This
process handles only controller/head pose data (WebSocket or Zenoh), converts
the controller poses into the robot convention, and emits the same EE chunk
contract used by the policy server. It never publishes ``/api/*``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import ssl
import threading
import time
from typing import Optional

import numpy as np

try:
    from . import topics
    from .action_chunk import ActionFrame, EndpointCommand, encode_action_chunk
    from .pose_math import apply_tool_rotation, change_basis, quaternion_to_matrix
    from .sec_dev_protocol import (
        DecodedSecDevPose,
        decode_sec_dev_pose,
        resolve_controller_analog,
    )
    from .xr_protocol import Button, InputFlag, make_haptic, parse_device_pose
except ImportError:
    import topics
    from action_chunk import ActionFrame, EndpointCommand, encode_action_chunk
    from pose_math import apply_tool_rotation, change_basis, quaternion_to_matrix
    from sec_dev_protocol import (
        DecodedSecDevPose,
        decode_sec_dev_pose,
        resolve_controller_analog,
    )
    from xr_protocol import Button, InputFlag, make_haptic, parse_device_pose


LOGGER = logging.getLogger("vr_teleop_server")

# Same WebXR-to-robot axis convention used by the previous TeleAvatar stack.
WEBXR_TO_ROBOT_BASIS = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
LEFT_TOOL_ROTATION = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
RIGHT_TOOL_ROTATION = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

# sec_dev field-6 input mask. The old WebXR packet uses a different Button enum.
SEC_LEFT_X = 0x01
SEC_LEFT_Y = 0x02
SEC_RIGHT_A = 0x04
SEC_RIGHT_B = 0x08


class VRTeleopNode:
    def __init__(
        self,
        *,
        wrist_offsets: dict[str, np.ndarray],
        gripper_input: str,
        ignore_tracking_flags: bool,
        control_hz: float,
        pose_transport: str,
        publish_debug_topics: bool,
    ):
        import rclpy
        from rclpy.qos import QoSProfile
        from std_msgs.msg import String

        self.rclpy = rclpy
        self.String = String
        self.node = rclpy.create_node("hil_vr_teleop_server")
        # Controller poses are latest-value data. A depth-one queue prevents
        # DDS from replaying old hand poses when the robot loop is busy.
        teleop_qos = QoSProfile(depth=1)
        self.wrist_offsets = wrist_offsets
        self.gripper_input = gripper_input
        self.ignore_tracking_flags = ignore_tracking_flags
        self.control_hz = float(control_hz)
        self.pose_transport = pose_transport
        self.publish_debug_topics = publish_debug_topics
        self._last_buttons = {"left": 0, "right": 0}
        self._last_input_mask: int | None = None
        self._sec_grippers = {"left": 0.0, "right": 0.0}
        self._sec_vr_announced = False
        self._chunk_sequence = 0
        self.status_lock = threading.Lock()
        self.pose_transport_connected = False
        self.tracking_valid = False
        self.last_pose_at = 0.0
        self.last_error = ""
        self.packets_received = 0
        self.packets_rejected = 0

        self.chunk_publisher = self.node.create_publisher(
            String, topics.TELEOP_ACTION_CHUNK, teleop_qos
        )
        self.mode_publisher = self.node.create_publisher(
            String, topics.MODE_REQUEST, QoSProfile(depth=1)
        )
        self.status_publisher = self.node.create_publisher(
            String, topics.TELEOP_SERVER_STATUS, 10
        )

        self.PoseStamped = None
        self.Joy = None
        self.raw_pose_publishers = {}
        self.input_publishers = {}
        if publish_debug_topics:
            from geometry_msgs.msg import PoseStamped
            from sensor_msgs.msg import Joy

            self.PoseStamped = PoseStamped
            self.Joy = Joy
            self.raw_pose_publishers = {
                "head": self.node.create_publisher(
                    PoseStamped, topics.VR_HEAD_POSE, 10
                ),
                "left": self.node.create_publisher(
                    PoseStamped, topics.VR_LEFT_CONTROLLER_POSE, 10
                ),
                "right": self.node.create_publisher(
                    PoseStamped, topics.VR_RIGHT_CONTROLLER_POSE, 10
                ),
            }
            self.input_publishers = {
                "left": self.node.create_publisher(Joy, topics.VR_LEFT_INPUT, 10),
                "right": self.node.create_publisher(Joy, topics.VR_RIGHT_INPUT, 10),
            }

        self.node.create_timer(0.2, self._publish_status)

        from rclpy.executors import SingleThreadedExecutor

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.spin_thread.start()

    def _spin_ros(self) -> None:
        from rclpy.executors import ExternalShutdownException

        try:
            self.executor.spin()
        except ExternalShutdownException:
            pass

    @staticmethod
    def _clamp_unit(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _pose_message(self, position, quaternion, frame_id: str):
        msg = self.PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(
            float, position
        )
        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = map(float, quaternion)
        return msg

    def _transform_pose(self, pose: dict, *, side: Optional[str] = None):
        position, quaternion = change_basis(
            pose["position"], pose["quaternion"], WEBXR_TO_ROBOT_BASIS
        )
        if side is not None:
            position = position - (
                quaternion_to_matrix(quaternion) @ self.wrist_offsets[side]
            )
            tool_rotation = (
                LEFT_TOOL_ROTATION if side == "left" else RIGHT_TOOL_ROTATION
            )
            quaternion = apply_tool_rotation(quaternion, tool_rotation)
        return position, quaternion

    def _controller_command(self, side: str, controller: dict) -> EndpointCommand:
        position, quaternion = self._transform_pose(
            controller["aim_pose"], side=side
        )
        return EndpointCommand(
            position=position,
            quaternion=quaternion,
            gripper=self._clamp_unit(controller[self.gripper_input]),
        )

    def _sec_controller_command(self, side: str, packet: DecodedSecDevPose) -> EndpointCommand:
        pose = packet.poses[side]
        position, quaternion = self._transform_pose(
            {"position": pose.position, "quaternion": pose.quaternion}, side=side
        )
        self._sec_grippers[side] = resolve_controller_analog(
            packet,
            side,
            self.gripper_input,
            self._sec_grippers[side],
            released_value=0.0,
        )
        return EndpointCommand(
            position=position,
            quaternion=quaternion,
            gripper=self._sec_grippers[side],
        )

    def _publish_debug(self, packet: dict) -> None:
        if not self.publish_debug_topics:
            return
        head_position, head_quaternion = self._transform_pose(packet["hmd"])
        self.raw_pose_publishers["head"].publish(
            self._pose_message(head_position, head_quaternion, "hil_vr_tracking")
        )
        button_order = (
            Button.A_CLICK,
            Button.B_CLICK,
            Button.X_CLICK,
            Button.Y_CLICK,
            Button.TRIGGER_CLICK,
            Button.SQUEEZE_CLICK,
            Button.THUMB_CLICK,
            Button.SYS_CLICK,
            Button.MENU_CLICK,
            Button.PINCH,
        )
        for side in ("left", "right"):
            controller = packet[side]
            position, quaternion = self._transform_pose(controller["aim_pose"])
            self.raw_pose_publishers[side].publish(
                self._pose_message(position, quaternion, "hil_vr_tracking")
            )
            joy = self.Joy()
            joy.header.stamp = self.node.get_clock().now().to_msg()
            joy.axes = [
                float(controller["stick"][0]),
                float(controller["stick"][1]),
                self._clamp_unit(controller["trigger"]),
                self._clamp_unit(controller["squeeze"]),
            ]
            joy.buttons = [
                int(bool(controller["buttons"] & button))
                for button in button_order
            ]
            self.input_publishers[side].publish(joy)

    def _publish_mode_edges(self, packet: dict) -> list[int]:
        haptic_controllers = []
        current = {
            "left": int(packet["left"]["buttons"]),
            "right": int(packet["right"]["buttons"]),
        }

        def rising(side: str, button: Button) -> bool:
            return bool(current[side] & button) and not bool(
                self._last_buttons[side] & button
            )

        requested = None
        if rising("right", Button.A_CLICK):
            requested = "teleop"
            haptic_controllers.append(1)
        elif rising("left", Button.X_CLICK):
            requested = "policy"
            haptic_controllers.append(0)
        elif rising("right", Button.B_CLICK) or rising("left", Button.Y_CLICK):
            requested = "pause"
            haptic_controllers.extend((0, 1))

        if requested is not None:
            self.mode_publisher.publish(self.String(data=requested))
            LOGGER.info("Requested HIL mode: %s", requested)
        self._last_buttons = current
        return haptic_controllers

    def _publish_sec_mode_edges(self, input_mask: int) -> None:
        """Translate sec_dev A/B/X/Y edges to HIL source requests."""
        previous = self._last_input_mask
        self._last_input_mask = int(input_mask)
        if previous is None:
            return
        rising = int(input_mask) & ~previous
        requested = None
        if rising & SEC_RIGHT_A:
            requested = "teleop"
        elif rising & SEC_LEFT_X:
            requested = "policy"
        elif rising & SEC_RIGHT_B or rising & SEC_LEFT_Y:
            requested = "pause"

        if requested is not None:
            self.mode_publisher.publish(self.String(data=requested))
            LOGGER.info("[VR] sec_dev request: %s", requested)

    def _publish_sec_debug(self, packet: DecodedSecDevPose) -> None:
        if not self.publish_debug_topics:
            return
        if "head" in packet.poses:
            head = packet.poses["head"]
            position, quaternion = self._transform_pose(
                {"position": head.position, "quaternion": head.quaternion}
            )
            self.raw_pose_publishers["head"].publish(
                self._pose_message(position, quaternion, "hil_vr_tracking")
            )
        for side in ("left", "right"):
            pose = packet.poses[side]
            position, quaternion = self._transform_pose(
                {"position": pose.position, "quaternion": pose.quaternion}
            )
            self.raw_pose_publishers[side].publish(
                self._pose_message(position, quaternion, "hil_vr_tracking")
            )
            joy = self.Joy()
            stick = packet.sticks.get(side, (0.0, 0.0))
            joy.header.stamp = self.node.get_clock().now().to_msg()
            joy.axes = [
                float(stick[0]),
                float(stick[1]),
                float(self._sec_grippers[side]),
                float(self._sec_grippers[side]),
            ]
            self.input_publishers[side].publish(joy)

    def publish_sec_dev_packet(self, packet: DecodedSecDevPose) -> None:
        """Publish a current sec_dev sample as one HIL teleop action."""
        now = time.monotonic()
        with self.status_lock:
            self.tracking_valid = True
            self.last_pose_at = now
            self.packets_received += 1
            if not self._sec_vr_announced:
                self._sec_vr_announced = True
                LOGGER.info(
                    "[VR 已接入] sec_dev 位姿流正常；"
                    "右 A 开始遥操，左 X 切回 policy，右 B/左 Y 暂停"
                )

        action = ActionFrame(
            left=self._sec_controller_command("left", packet),
            right=self._sec_controller_command("right", packet),
        )
        self._chunk_sequence += 1
        payload = encode_action_chunk(
            source="teleop",
            sequence=self._chunk_sequence,
            control_hz=self.control_hz,
            frame_ids=topics.ACTION_FRAME_IDS["teleop"],
            actions=[action],
        )
        self.chunk_publisher.publish(self.String(data=payload))
        self._publish_sec_debug(packet)
        self._publish_sec_mode_edges(packet.input_mask)

    def publish_packet(self, packet: dict) -> list[int]:
        tracking_valid = self.ignore_tracking_flags or all(
            int(packet[side]["flags"]) & int(InputFlag.ENABLE)
            for side in ("left", "right")
        )
        now = time.monotonic()
        with self.status_lock:
            self.tracking_valid = tracking_valid
            self.packets_received += 1
            if tracking_valid:
                self.last_pose_at = now
        if not tracking_valid:
            return []

        action = ActionFrame(
            left=self._controller_command("left", packet["left"]),
            right=self._controller_command("right", packet["right"]),
        )
        self._chunk_sequence += 1
        payload = encode_action_chunk(
            source="teleop",
            sequence=self._chunk_sequence,
            control_hz=self.control_hz,
            frame_ids=topics.ACTION_FRAME_IDS["teleop"],
            actions=[action],
        )
        self.chunk_publisher.publish(self.String(data=payload))
        self._publish_debug(packet)
        return self._publish_mode_edges(packet)

    def reject_packet(self, error: Exception) -> None:
        with self.status_lock:
            self.packets_rejected += 1
            self.last_error = str(error)

    def set_pose_connected(self, connected: bool) -> None:
        with self.status_lock:
            self.pose_transport_connected = bool(connected)
            if connected:
                self.last_error = ""
            else:
                self.tracking_valid = False
                self._sec_vr_announced = False
                self._last_input_mask = None

    def _publish_status(self) -> None:
        now = time.monotonic()
        with self.status_lock:
            status = {
                "pose_transport": self.pose_transport,
                "pose_protocol": "sec_dev" if self.pose_transport == "zenoh" else "webxr",
                "pose_transport_connected": self.pose_transport_connected,
                "tracking_valid": self.tracking_valid,
                "pose_age_s": (
                    None
                    if self.last_pose_at <= 0.0
                    else round(now - self.last_pose_at, 3)
                ),
                "chunk_sequence": self._chunk_sequence,
                "packets_received": self.packets_received,
                "packets_rejected": self.packets_rejected,
                "last_error": self.last_error,
                "video_transport": "native_second_s100_to_headset",
                "video_processed_by_host": False,
                "debug_ros_topics": self.publish_debug_topics,
            }
        self.status_publisher.publish(
            self.String(data=json.dumps(status, sort_keys=True))
        )

    def close(self) -> None:
        try:
            self.executor.shutdown()
        except Exception as exc:
            LOGGER.warning("ROS executor cleanup skipped: %s", exc)
        try:
            self.node.destroy_node()
        except Exception as exc:
            LOGGER.warning("VR teleop node cleanup skipped: %s", exc)
        try:
            if self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception as exc:
            LOGGER.warning("ROS context cleanup skipped: %s", exc)
        self.spin_thread.join(timeout=1.0)


async def run_websocket(node: VRTeleopNode, args) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Install websockets before using WebSocket pose input") from exc

    ssl_context = None
    if args.uri.startswith("wss://") and args.insecure_tls:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    while node.rclpy.ok():
        try:
            LOGGER.info("Connecting to VR pose WebSocket: %s", args.uri)
            async with websockets.connect(
                args.uri,
                ssl=ssl_context,
                ping_interval=10.0,
                ping_timeout=5.0,
                max_size=4096,
            ) as websocket:
                node.set_pose_connected(True)
                try:
                    async for message in websocket:
                        if not isinstance(message, bytes):
                            continue
                        try:
                            packet = parse_device_pose(message)
                            haptics = node.publish_packet(packet)
                        except ValueError as exc:
                            node.reject_packet(exc)
                            continue
                        for controller_index in haptics:
                            await websocket.send(
                                make_haptic(
                                    controller_index, duration_ms=80, strength=0.4
                                )
                            )
                finally:
                    node.set_pose_connected(False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            node.reject_packet(exc)
            LOGGER.warning(
                "VR pose connection lost: %s; retrying in %.1fs",
                exc,
                args.reconnect_delay,
            )
            await asyncio.sleep(args.reconnect_delay)


def _zenoh_payload_bytes(payload) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if hasattr(payload, "to_bytes"):
        return payload.to_bytes()
    return bytes(payload)


async def run_zenoh(node: VRTeleopNode, args) -> None:
    try:
        import zenoh
    except ImportError as exc:
        # Allow ROS Python to reuse an ABI-compatible Zenoh wheel installed in
        # another environment without embedding a machine-specific path.
        import os
        import sys

        fallback = os.environ.get("HIL_ZENOH_SITE_PACKAGES", "")
        if fallback and os.path.isdir(fallback) and fallback not in sys.path:
            sys.path.append(fallback)
        try:
            import zenoh
        except ImportError:
            raise RuntimeError(
                "Install eclipse-zenoh for the ROS Python interpreter, or set "
                "HIL_ZENOH_SITE_PACKAGES to a directory containing the abi3 zenoh wheel"
            ) from exc

    config = zenoh.Config()
    if args.zenoh_endpoint:
        config.insert_json5("connect/endpoints", json.dumps([args.zenoh_endpoint]))
    session = zenoh.open(config)
    loop = asyncio.get_running_loop()
    packets: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)

    def enqueue_latest(data: bytes) -> None:
        if packets.full():
            packets.get_nowait()
        packets.put_nowait(data)

    def on_sample(sample) -> None:
        try:
            data = _zenoh_payload_bytes(sample.payload)
            loop.call_soon_threadsafe(enqueue_latest, data)
        except Exception as exc:
            node.reject_packet(exc)

    subscriber = session.declare_subscriber(args.zenoh_key, on_sample)
    node.set_pose_connected(True)
    LOGGER.info(
        "Subscribed to Zenoh VR pose key %s endpoint=%s",
        args.zenoh_key,
        args.zenoh_endpoint or "auto-discovery",
    )
    try:
        while node.rclpy.ok():
            try:
                data = await asyncio.wait_for(packets.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                node.publish_sec_dev_packet(decode_sec_dev_pose(data))
            except ValueError as exc:
                node.reject_packet(exc)
    finally:
        node.set_pose_connected(False)
        try:
            if hasattr(subscriber, "undeclare"):
                subscriber.undeclare()
        except Exception as exc:
            LOGGER.warning("Zenoh subscriber cleanup failed: %s", exc)
        try:
            session.close()
        except Exception as exc:
            LOGGER.warning("Zenoh session cleanup failed: %s", exc)


def parse_offset(text: str) -> np.ndarray:
    values = np.asarray([float(value) for value in text.split(",")], dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("Expected three comma-separated finite values")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose-transport", choices=("websocket", "zenoh"), default="zenoh"
    )
    parser.add_argument("--uri", default="ws://127.0.0.1:8080/devicepose")
    parser.add_argument("--zenoh-endpoint", default="")
    parser.add_argument("--zenoh-key", default=topics.VR_ZENOH_DEVICEPOSE_KEY)
    parser.add_argument("--insecure-tls", action="store_true")
    parser.add_argument("--reconnect-delay", type=float, default=1.0)
    parser.add_argument("--control-hz", type=float, default=60.0)
    parser.add_argument(
        "--ignore-tracking-flags",
        action="store_true",
        help="Accept packets even when a controller ENABLE flag is absent.",
    )
    parser.add_argument(
        "--publish-debug-topics",
        action="store_true",
        help="Optionally expose raw head/controller PoseStamped and Joy topics.",
    )
    parser.add_argument(
        "--gripper-input",
        choices=("trigger", "squeeze"),
        default="trigger",
    )
    parser.add_argument(
        "--left-wrist-offset",
        type=parse_offset,
        default=np.array([0.13205294, -0.03416498, 0.03397751]),
    )
    parser.add_argument(
        "--right-wrist-offset",
        type=parse_offset,
        default=np.array([0.12516124, 0.04039848, 0.03931866]),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.reconnect_delay <= 0.0 or args.control_hz <= 0.0:
        parser.error("--reconnect-delay and --control-hz must be greater than zero")
    if not args.zenoh_key:
        parser.error("--zenoh-key must not be empty")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    import rclpy

    rclpy.init()
    node = VRTeleopNode(
        wrist_offsets={
            "left": np.asarray(args.left_wrist_offset, dtype=np.float64),
            "right": np.asarray(args.right_wrist_offset, dtype=np.float64),
        },
        gripper_input=args.gripper_input,
        ignore_tracking_flags=args.ignore_tracking_flags,
        control_hz=args.control_hz,
        pose_transport=args.pose_transport,
        publish_debug_topics=args.publish_debug_topics,
    )
    try:
        runner = run_websocket if args.pose_transport == "websocket" else run_zenoh
        asyncio.run(runner(node, args))
    except KeyboardInterrupt:
        pass
    finally:
        node.close()


if __name__ == "__main__":
    main()
