#!/usr/bin/env python3
"""Environment for TA2 relative end-effector inference."""

from examples.teleavatar_v2 import ros2_interface_relative_endeffector
from examples.teleavatar_v2.env_endeffector import TeleavatarEndEffectorEnvironment


class TeleavatarRelativeEndEffectorEnvironment(TeleavatarEndEffectorEnvironment):
    """Use UMI's 16-D state (EE poses plus commanded gripper values)."""

    ros_interface_class = ros2_interface_relative_endeffector.TeleavatarRelativeEndEffectorROS2Interface
