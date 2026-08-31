#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32
import numpy as np


def quaternion_slerp(q0, q1, t, shortest=True):
    q0 = np.array(q0, dtype=np.float64)
    q1 = np.array(q1, dtype=np.float64)

    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)

    dot = np.dot(q0, q1)
    if shortest and dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)

    if np.abs(dot - 1.0) < 1e-6:
        return q0.tolist()

    theta = np.arccos(dot) * t
    sin_theta = np.sin(np.arccos(dot))
    q_rot = q0 * np.cos(theta) + (q1 - q0 * dot) * np.sin(theta) / sin_theta
    return q_rot.tolist()


def make_pose(px, py, pz, qx, qy, qz, qw):
    pose = Pose()
    pose.position.x = px
    pose.position.y = py
    pose.position.z = pz
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


# 夹爪指令为力控（非开合度）：0.0 = 张开方向最大力，1.0 = 合爪方向最大力，
# 0.10 = 前馈力矩零点。归零取 0.0 让夹爪完全张开。
GRIPPER_ZERO_CMD = 0.0

# 双臂到达后继续保持受控、等待夹爪动作完成的时间（秒）。
# 夹爪指令随手臂指令同路下发，失能后夹爪不再响应，因此需在失能前留出动作窗口。
GRIPPER_ACTION_DURATION = 3.0

# 双臂末端归零位置取完整 floor_2_for_ee 转换数据集 174 个 episode 的
# 第 0 帧 observation.state 平均值。姿态沿用已验证的中性末端朝向。
ZERO_POSES = {
    "left_arm": make_pose(
        0.234892075346119, -0.012588418568609, -0.364043021852943,
        -0.707106781186548, 0.0, 0.0, 0.707106781186548,
    ),
    "right_arm": make_pose(
        0.179305963089754, -0.044530757185709, -0.366983165001047,
        0.707106781186548, 0.0, 0.0, 0.707106781186548,
    ),
}


class ArmMover:
    """单条手臂的轨迹规划与到达检测（由 DualArmZeroNode 统一驱动）"""

    def __init__(self, node: Node, namespace: str, target_pose: Pose):
        self.node = node
        self.namespace = namespace
        self.target_pose = target_pose

        self.pose_pub = node.create_publisher(
            Pose, f"/api/{namespace}/target_pose", 10)
        self.pose_sub = node.create_subscription(
            Pose, f"/{namespace}/current_ee_pose", self.pose_callback, 10)

        side = namespace.split("_")[0]  # left_arm -> left
        self.gripper_pub = node.create_publisher(
            Float32, f"/api/{side}_gripper/cmd", 10)

        self.current_pose = None
        self.traj_points = []
        self.traj_index = 0
        self.plan_done = False
        self.reached = False

        self.pos_tolerance = 0.005
        self.quat_dot_tolerance = 0.99

    def pose_callback(self, msg):
        self.current_pose = msg

        if not self.plan_done:
            self.generate_trajectory()
            self.plan_done = True

        if not self.reached:
            self.check_reached()

    def generate_trajectory(self):
        start = self.current_pose
        p_start = [start.position.x, start.position.y, start.position.z]
        q_start = [start.orientation.x, start.orientation.y,
                   start.orientation.z, start.orientation.w]

        p_target = [self.target_pose.position.x, self.target_pose.position.y,
                    self.target_pose.position.z]
        q_target = [self.target_pose.orientation.x, self.target_pose.orientation.y,
                    self.target_pose.orientation.z, self.target_pose.orientation.w]

        # 根据直线距离决定插值步数（至少 20 步）
        dist = np.linalg.norm(np.array(p_target) - np.array(p_start))
        steps = max(int(dist / 0.01), 20)

        self.traj_points = []
        for i in range(steps + 1):
            t = i / steps

            px = p_start[0] + (p_target[0] - p_start[0]) * t
            py = p_start[1] + (p_target[1] - p_start[1]) * t
            pz = p_start[2] + (p_target[2] - p_start[2]) * t

            q_interp = quaternion_slerp(q_start, q_target, t)

            self.traj_points.append(make_pose(px, py, pz, *q_interp))

        self.traj_index = 0
        self.node.get_logger().info(
            f"[{self.namespace}] Trajectory generated with {len(self.traj_points)} points")

    def publish_command(self):
        """发布手臂指令：轨迹阶段逐点发布，之后持续回发目标位姿。

        夹爪指令随手臂指令同路下发，手臂指令流一旦停发，夹爪也不会动作，
        因此轨迹发完后必须继续回发目标位姿保持受控。
        """
        if not self.plan_done:
            return

        if self.traj_index < len(self.traj_points):
            self.pose_pub.publish(self.traj_points[self.traj_index])
            self.traj_index += 1
            if self.traj_index == len(self.traj_points):
                self.node.get_logger().info(
                    f"[{self.namespace}] All trajectory points published, "
                    f"holding target pose until arrival...")
        else:
            self.pose_pub.publish(self.target_pose)

    def check_reached(self):
        dx = self.current_pose.position.x - self.target_pose.position.x
        dy = self.current_pose.position.y - self.target_pose.position.y
        dz = self.current_pose.position.z - self.target_pose.position.z
        pos_err = np.sqrt(dx * dx + dy * dy + dz * dz)

        q_curr = [self.current_pose.orientation.x, self.current_pose.orientation.y,
                  self.current_pose.orientation.z, self.current_pose.orientation.w]
        q_tar = [self.target_pose.orientation.x, self.target_pose.orientation.y,
                 self.target_pose.orientation.z, self.target_pose.orientation.w]

        q_curr = np.array(q_curr) / np.linalg.norm(q_curr)
        q_tar = np.array(q_tar) / np.linalg.norm(q_tar)
        dot = abs(np.dot(q_curr, q_tar))

        if pos_err < self.pos_tolerance and dot > self.quat_dot_tolerance:
            self.reached = True
            self.node.get_logger().info(
                f"[{self.namespace}] Target reached! Position error = {pos_err:.4f} m, "
                f"Quaternion dot = {dot:.4f}.")


class DualArmZeroNode(Node):
    def __init__(self):
        super().__init__("dual_arm_zero_node")

        self.enable_pub = self.create_publisher(Float32, "/api/fsm/enable", 10)

        self.arms = [
            ArmMover(self, namespace, target)
            for namespace, target in ZERO_POSES.items()
        ]

        self.enable_timer = self.create_timer(0.05, self.enable_callback)
        self.traj_timer = self.create_timer(0.02, self.timer_callback)

        self.reached_time = None
        self.finished = False

    def enable_callback(self):
        msg = Float32()
        msg.data = 1.0
        self.enable_pub.publish(msg)

        gripper_msg = Float32()
        gripper_msg.data = GRIPPER_ZERO_CMD
        for arm in self.arms:
            arm.gripper_pub.publish(gripper_msg)

    def timer_callback(self):
        if self.finished:
            return

        # 全程保持手臂指令流（含到达后的保持阶段），夹爪才会响应
        for arm in self.arms:
            arm.publish_command()

        if self.reached_time is None:
            if all(arm.reached for arm in self.arms):
                self.reached_time = self.get_clock().now()
                self.get_logger().info(
                    f"Both arms reached zero pose. Holding for "
                    f"{GRIPPER_ACTION_DURATION}s to let grippers act...")
            return

        elapsed = (self.get_clock().now() - self.reached_time).nanoseconds / 1e9
        if elapsed >= GRIPPER_ACTION_DURATION:
            self.finish()

    def finish(self):
        self.finished = True
        self.traj_timer.cancel()
        self.enable_timer.cancel()
        disable_msg = Float32()
        disable_msg.data = 0.0
        self.enable_pub.publish(disable_msg)
        self.get_logger().info(
            "Gripper action window elapsed. Disable signal sent, timers stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = DualArmZeroNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
