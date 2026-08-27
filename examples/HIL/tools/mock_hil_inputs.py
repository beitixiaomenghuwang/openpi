#!/usr/bin/env python3
"""Publish stationary fake robot/teleop inputs for supervisor dry-run tests."""

from __future__ import annotations

import argparse
import json

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, String

from hil import topics
from hil.action_chunk import ActionFrame, EndpointCommand, encode_action_chunk


def make_pose(x: float, y: float, z: float) -> Pose:
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.w = 1.0
    return pose


class MockInputs(Node):
    def __init__(self, requested_mode: str):
        super().__init__("hil_mock_inputs")
        self.requested_mode = requested_mode
        self.fsm_state = 0
        self.ticks = 0
        self.robot_poses = {
            "left": make_pose(0.10, 0.15, -0.42),
            "right": make_pose(0.10, -0.15, -0.42),
        }
        self.mode_pub = self.create_publisher(String, topics.ROBOT_CURRENT_MODE, 10)
        self.fsm_pub = self.create_publisher(Int32, topics.ROBOT_FSM_STATE, 10)
        self.robot_pose_pubs = {
            "left": self.create_publisher(Pose, topics.ROBOT_LEFT_POSE, 10),
            "right": self.create_publisher(Pose, topics.ROBOT_RIGHT_POSE, 10),
        }
        self.teleop_chunk_pub = self.create_publisher(
            String, topics.TELEOP_ACTION_CHUNK, 10
        )
        self.request_pub = self.create_publisher(String, topics.MODE_REQUEST, 10)
        self.create_subscription(Float32, topics.API_FSM_ENABLE, self._on_enable, 10)
        self.create_timer(0.02, self._tick)

    def _on_enable(self, msg) -> None:
        self.fsm_state = 2 if msg.data >= 0.5 else 0

    def _tick(self) -> None:
        self.ticks += 1
        mode = {
            "meta_mode": 1,
            "left_arm_control_mode": 0,
            "right_arm_control_mode": 0,
            "enable_left_arm": True,
            "enable_right_arm": True,
        }
        self.mode_pub.publish(String(data=json.dumps(mode)))
        self.fsm_pub.publish(Int32(data=self.fsm_state))
        for side in ("left", "right"):
            self.robot_pose_pubs[side].publish(self.robot_poses[side])
        position = (0.0002 * max(0, self.ticks - 100), 0.0, 0.0)
        command = EndpointCommand(position, (0.0, 0.0, 0.0, 1.0), 0.5)
        action = ActionFrame(left=command, right=command)
        self.teleop_chunk_pub.publish(
            String(
                data=encode_action_chunk(
                    source="teleop",
                    sequence=self.ticks,
                    control_hz=50.0,
                    frame_ids=topics.ACTION_FRAME_IDS["teleop"],
                    actions=[action],
                )
            )
        )
        if self.ticks == 75:
            self.request_pub.publish(String(data=self.requested_mode))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("teleop", "pause"), default="teleop")
    args = parser.parse_args()
    rclpy.init()
    node = MockInputs(args.mode)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
