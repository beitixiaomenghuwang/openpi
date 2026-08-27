"""ROS topic contract for the minimal API-endpoint HIL runtime.

The policy adapter and VR teleop node publish action chunks. Only
``hil_supervisor.py`` may publish topics below ``/api``.
"""

HIL_NS = "/hil"

# Pose packets published by the second S100/VR middleware. Keep this key in
# one place so the runtime subscriber and diagnostics cannot silently diverge.
VR_ZENOH_DEVICEPOSE_KEY = "sec/xr/devicepose"

MODE_REQUEST = f"{HIL_NS}/control/mode_request"
MODE_STATUS = f"{HIL_NS}/control/status"

POLICY_INPUT_ACTION_CHUNK = f"{HIL_NS}/policy/input/ee_action_chunk"
# Legacy end-effector policy inputs. The policy interface publishes these
# four values; policy_topic_adapter.py combines one fresh value from each
# stream into POLICY_INPUT_ACTION_CHUNK.
POLICY_INPUT_LEFT_POSE = f"{HIL_NS}/policy/input/left_arm/target_pose"
POLICY_INPUT_RIGHT_POSE = f"{HIL_NS}/policy/input/right_arm/target_pose"
POLICY_INPUT_LEFT_GRIPPER = f"{HIL_NS}/policy/input/left_gripper/cmd"
POLICY_INPUT_RIGHT_GRIPPER = f"{HIL_NS}/policy/input/right_gripper/cmd"
TELEOP_SERVER_STATUS = f"{HIL_NS}/teleop/status"

# The supervisor consumes both source streams and is the sole owner of the
# robot command topics. Policy is intentionally a direct external input.
TELEOP_ACTION_CHUNK = f"{HIL_NS}/source/teleop/ee_action_chunk"

VR_HEAD_POSE = f"{HIL_NS}/input/vr/head_pose"
VR_LEFT_CONTROLLER_POSE = f"{HIL_NS}/input/vr/left_controller_pose"
VR_RIGHT_CONTROLLER_POSE = f"{HIL_NS}/input/vr/right_controller_pose"
VR_LEFT_INPUT = f"{HIL_NS}/input/vr/left_controller"
VR_RIGHT_INPUT = f"{HIL_NS}/input/vr/right_controller"

API_LEFT_POSE = "/api/left_arm/target_pose"
API_RIGHT_POSE = "/api/right_arm/target_pose"
API_LEFT_GRIPPER = "/api/left_gripper/cmd"
API_RIGHT_GRIPPER = "/api/right_gripper/cmd"
API_FSM_ENABLE = "/api/fsm/enable"

ROBOT_LEFT_POSE = "/left_arm/current_ee_pose"
ROBOT_RIGHT_POSE = "/right_arm/current_ee_pose"
ROBOT_FSM_STATE = "/fsm_state"
ROBOT_CURRENT_MODE = "/api/current_mode"

MODES = ("pause", "policy", "teleop")

ACTION_FRAME_IDS = {
    "policy": {
        "left": "left_shoulder_base",
        "right": "right_shoulder_base",
    },
    "teleop": {
        "left": "hil_teleop_left_relative",
        "right": "hil_teleop_right_relative",
    },
}


def action_chunk_topic(source: str) -> str:
    if source == "policy":
        return POLICY_INPUT_ACTION_CHUNK
    if source == "teleop":
        return TELEOP_ACTION_CHUNK
    raise ValueError(f"Unknown HIL source: {source}")
