# TeleAvatar 2 Minimal HIL Router

This OpenPI example contains only the API endpoint-mode HIL control boundary. An
external policy owns its image/state subscriptions and inference runtime. The
HIL runtime owns VR teleoperation, source switching, policy-input validation,
and the only publishers allowed on the robot `/api/*` command topics.

This directory currently runs as a standalone ROS 2 process group. It has not
yet been integrated with OpenPI's policy server/client runtime.

Required remoteApp configuration:

- `meta_mode = 1` (API)
- `left_arm_control_mode = 0` (endpoint)
- `right_arm_control_mode = 0` (endpoint)
- both arms enabled

## Architecture

```text
external policy
  raw images + robot state -> inference
  four /hil/policy/input/* topics -> policy_topic_adapter
  /hil/policy/input/ee_action_chunk
                         \
                          +--> hil_supervisor --> /api/* --> Robot
                         /
second S100/VR -> vr_teleop_server -> teleop action chunk
```

There is no model-video decoder, fake policy, or policy gateway in this
example. The external policy may subscribe to any original ROS topics or
use its own server/client transport. It must never publish `/api/*` while HIL
is running.

## Runtime Files

- `hil/hil_supervisor.py`: validates both action sources, performs switching,
  limits teleop motion, drives the API FSM, and exclusively publishes `/api/*`
  in the HIL runtime.
- `hil/policy_topic_adapter.py`: synchronizes the policy's two endpoint poses
  and two gripper commands, then emits one canonical policy action chunk.
- `hil/vr_teleop_server.py`: decodes the second S100 controller stream and
  publishes relative dual-arm teleop chunks.
- `hil/action_chunk.py`: canonical external-policy JSON encoder/decoder.
- `hil/sec_dev_protocol.py`, `hil/zenoh_wire.py`, `hil/xr_protocol.py`,
  `hil/pose_math.py`: protocol and pose primitives used by teleoperation.
- `scripts/run_hil.sh`: starts the policy adapter, VR teleoperation, and the
  Supervisor.
- `tools/`: standalone diagnostics and manual API tools; they are not started
  by `run_hil.sh` and are outside the HIL ownership boundary. Never run an API
  tool at the same time as the HIL Supervisor.

## ROS Contract

| Direction | Topic | Type |
| --- | --- | --- |
| external policy -> adapter | `/hil/policy/input/left_arm/target_pose` | `Pose` |
| external policy -> adapter | `/hil/policy/input/right_arm/target_pose` | `Pose` |
| external policy -> adapter | `/hil/policy/input/left_gripper/cmd` | `Float32` |
| external policy -> adapter | `/hil/policy/input/right_gripper/cmd` | `Float32` |
| adapter -> supervisor | `/hil/policy/input/ee_action_chunk` | JSON `std_msgs/String` |
| VR teleop -> supervisor | `/hil/source/teleop/ee_action_chunk` | JSON `std_msgs/String` |
| VR buttons -> supervisor | `/hil/control/mode_request` | `String`: `policy`, `teleop`, `pause` |
| supervisor -> policy/observers | `/hil/control/status` | JSON `std_msgs/String` |
| supervisor -> robot | `/api/{left,right}_arm/target_pose` | `Pose` |
| supervisor -> robot | `/api/{left,right}_gripper/cmd` | `Float32` |
| supervisor -> robot | `/api/fsm/enable` | `Float32` |

Policy actions are absolute endpoint poses in `left_shoulder_base` and
`right_shoulder_base`. A chunk contains synchronized left/right actions with
`position`, xyzw `quaternion`, and normalized gripper command in `[0, 1]`.
Both policy and teleop chunks must contain exactly one action frame.
The schema is `teleavatar.hil.ee_action_chunk.v1`.

## External Policy Adapter

The policy may obtain all observations directly. Its interface publishes the
same two `Pose` and two `Float32` values as before, but uses the four topics
listed above instead of `/api/*`. It must not publish `/api/fsm/enable`.
Only the publisher topic names need to change; no policy-side HIL state machine
is required.

```python
from hil import topics

self.action_publishers = {
    "left_ee": self.create_publisher(Pose, topics.POLICY_INPUT_LEFT_POSE, 10),
    "right_ee": self.create_publisher(Pose, topics.POLICY_INPUT_RIGHT_POSE, 10),
    "left_gripper": self.create_publisher(
        Float32, topics.POLICY_INPUT_LEFT_GRIPPER, 10
    ),
    "right_gripper": self.create_publisher(
        Float32, topics.POLICY_INPUT_RIGHT_GRIPPER, 10
    ),
}
```

The adapter requires all four topics to update before every output and rejects
a set whose receive-time spread exceeds `HIL_POLICY_SYNC_SLOP` (default
`0.02` seconds). The callback carrying the fourth fresh value publishes the
single-action chunk immediately; there is no polling timer. It does not expose
any `/api/*` publisher. The policy does not need to subscribe to HIL status,
stop inference, clear its own action cache, or implement handback logic.
`HIL_POLICY_CONTROL_HZ` is chunk metadata only; it does not throttle or resample
the policy stream. The actual output cadence follows complete four-topic input
sets one-for-one.

The policy interface is hardcoded to the four `/hil/policy/input/*` names. The
robot only subscribes to the `/api/*` names published by the Supervisor, so the
four input messages cannot be consumed by the robot directly.

## Policy Handback

Every policy request deletes the previously cached policy action. The Supervisor
keeps the robot in FSM `PAUSE` and waits for the next complete four-topic action
received after that request. It still requires the expected dual-arm endpoint
frames, finite normalized values, and a fresh source timestamp. At handback
the Supervisor does not reject a target because it differs from current endpoint
feedback. After FSM `READY`, it follows the newest policy target at 60 Hz using
the configured `--max-translation-speed` and `--max-rotation-speed`. Once the
command has reached that target and measured endpoint feedback is within
`0.03 m` and `0.262 rad`, it leaves the handback transition and resumes direct
policy forwarding. These completion tolerances can be changed with
`--policy-handover-position-limit` and `--policy-handover-rotation-limit`.

The legacy four-topic protocol carries one action at a time and has no original
inference-generation identifier. HIL can therefore discard everything it
received before the policy request, but cannot tell whether the first action
after the request belongs to a rollout computed before or during teleoperation.
Guaranteeing a post-intervention inference requires a small policy-side reset or
generation identifier in a future protocol revision.

While policy is active, every accepted single-action chunk is forwarded to
`/api/*` immediately and exactly once. The Supervisor does not resample policy
output at its 60 Hz teleop rate and does not apply translation, rotation, reach,
or gripper slew limits to policy commands. Schema validation, mode/FSM checks,
source freshness, API publisher ownership, handback cache clearing, and the
one-time bounded handback transition still apply. Teleoperation retains its own
60 Hz motion shaping and limits. It stores
only the newest VR target and keeps moving toward that target on every control
tick until a newer sample replaces it; headset samples are never accumulated as
a trajectory queue.

Relevant diagnostic status fields include `phase`, `pending_mode`,
`source_action_age_s`, `source_queue_depth`, `source_accepted_chunks`,
`source_rejected_chunks`, and `source_last_rejection`. For backward-compatible
status consumers, `source_queue_depth` is now `0` or `1` and indicates whether a
latest teleop target is buffered; it is not a FIFO depth. The external policy
does not need these fields for normal operation.

## VR Controls

- Right A: request teleoperation.
- Left X: clear the old policy input, then smoothly track the next policy target.
- Right B or Left Y: pause and lock the robot.
- The selected trigger controls each gripper.

The default Zenoh key is `sec/xr/devicepose`. Auto-discovery is used when
`VR_ZENOH_ENDPOINT` is empty.

## Run

```bash
cd /path/to/openpi
source /opt/ros/humble/setup.zsh
export ROS_DOMAIN_ID=29
export HIL_PYTHON_BIN=/usr/bin/python3
bash examples/HIL/scripts/run_hil.sh
```

Install `eclipse-zenoh` for `HIL_PYTHON_BIN`. If an ABI-compatible Zenoh wheel
is installed in another environment, provide its package directory explicitly:

```bash
export HIL_ZENOH_SITE_PACKAGES=/path/to/python/site-packages
```

The default is dry-run and creates no `/api/*` output. Dry-run internally
completes the PAUSE/READY transition because it deliberately does not publish
`/api/fsm/enable`; current pose, mode, source, and ownership checks still run.
Enable robot commands only after checking ownership and clearing the workspace:

```bash
export HIL_ENABLE_API_OUTPUT=1
bash scripts/run_hil.sh
```

Before enabling output, verify that no other policy, standalone teleop script,
or robot command node publishes `/api/*`:

```bash
ros2 topic echo /api/current_mode --once
ros2 topic echo /hil/control/status
ros2 topic echo /hil/teleop/status
ros2 topic info --verbose /api/fsm/enable
```

`/api/fsm/enable` must show only `hil_api_endpoint_supervisor` as publisher.

## Checks

```bash
cd /path/to/openpi/examples/HIL
python3 -m unittest discover -s tests -v
python3 -m compileall -q hil tools tests
bash -n scripts/run_hil.sh
```

For a teleop-only ROS state-machine dry run, use an isolated domain. The mock
publishes fake robot state and must never run on the real robot domain:

```bash
ROS_DOMAIN_ID=230 python3 -m hil.hil_supervisor --dry-run \
  --skip-ownership-check --skip-teleop-status-check
ROS_DOMAIN_ID=230 python3 -m tools.mock_hil_inputs
```
