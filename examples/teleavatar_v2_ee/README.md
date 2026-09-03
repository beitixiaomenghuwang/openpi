# TeleAvatar V2 Bimanual End-Effector Deployment

This example deploys an OpenPI end-effector policy directly on both TeleAvatar
V2 arms. It is independent from the joint-space `examples/teleavatar_v2`
example and from HIL: it does not import HIL code or publish `/hil/*` topics.

The matching training configs are:

- `pi0_teleavatar_v2_ee`
- `pi05_teleavatar_v2_ee`
- `pi05_teleavatar_v2_ee_rtc` (RTC `TRAINED` mode)
- `pi0_teleavatar_v2_ee_low_mem_finetune`

Each checkpoint returns a synchronized 20D action chunk:
`[left EE 10D, right EE 10D]`, where each arm is
`[position(3), row-wise rotation-6D(6), gripper trigger(1)]`.

## Data Flow

```text
S100 RTP/H.265 composite -> three split RGB views --+
                                                     +-> WebSocket policy server
/<arm>_arm/current_ee_pose ------------------------->+          |
                                                                v
                               absolute pose9 + trigger action chunk
                                                                |
                                                                v
                    /api/<arm>_arm/target_pose
                    /api/<arm>_gripper/cmd
                    /api/fsm/enable
```

Both arms receive one pose and one gripper command from every action step.

## Robot Setup

1. Put both arms in API mode with end-effector control enabled.
2. Point the S100 RTP remote IP at the client machine (UDP port 8890 by default).
3. Connect the client to the robot's ROS2 network. For example:

   ```bash
   export ROS_DOMAIN_ID=29
   export ROS_DISTRO=humble
   zenoh-bridge-ros2dds -e tcp/<ROBOT_IP>:9000
   ```

4. Verify both current-pose topics before deployment:

   ```bash
   export ROS_DOMAIN_ID=29
   ros2 topic echo /left_arm/current_ee_pose --once
   ros2 topic echo /right_arm/current_ee_pose --once
   ```

Do not run another program that publishes the same `/api/*` topics.

## Start the Policy Server

Run on the inference machine, using the config that produced the checkpoint:

```bash
cd /path/to/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_teleavatar_v2_ee \
  --policy.dir=checkpoints/pi0_teleavatar_v2_ee/<experiment>/<step>
```

The checkpoint must contain its `assets/<dataset>/norm_stats.json` and must be
trained with the bimanual EE config.

For the first offline or simulation validation, enable RTC on the server and
use the training-free `GUIDED` mode:

```bash
uv run scripts/serve_policy.py \
  --rtc.enabled --rtc.mode=GUIDED \
  policy:checkpoint \
  --policy.config=pi05_teleavatar_v2_ee \
  --policy.dir=checkpoints/pi05_teleavatar_v2_ee/<experiment>/<step>
```

After `pi05_teleavatar_v2_ee_rtc` has been trained and the measured delay is
confirmed stable, switch only the server mode and checkpoint to `TRAINED`:

```bash
uv run scripts/serve_policy.py \
  --rtc.enabled --rtc.mode=TRAINED \
  policy:checkpoint \
  --policy.config=pi05_teleavatar_v2_ee_rtc \
  --policy.dir=checkpoints/pi05_teleavatar_v2_ee_rtc/<experiment>/<step>
```

## Run the Client

First run read-only. This receives live observations, calls the policy, checks
the returned `(N, 20)` bimanual chunks, and prints their first target without moving the
robot:

```bash
cd /path/to/openpi
source /opt/ros/humble/setup.zsh
export ROS_DOMAIN_ID=29
python examples/teleavatar_v2_ee/main.py \
  --remote-host <POLICY_SERVER_IP> \
  --prompt "perform the manipulation task" \
  --max-steps 90
```

Add `--rtc` when connecting to an RTC-enabled server. The broker calibrates
the round-trip delay at startup, then requests one new chunk asynchronously
while this loop consumes exactly one action per control tick:

```bash
python examples/teleavatar_v2_ee/main.py \
  --remote-host <POLICY_SERVER_IP> \
  --prompt "perform the manipulation task" \
  --rtc --max-steps 90
```

Keep this command read-only until the GUIDED latency and continuity checks
pass; add `--publish` only for robot control.

For real control, add `--publish`. After the sensor and policy-server checks
complete, inference and command publication start automatically:

```bash
python examples/teleavatar_v2_ee/main.py \
  --remote-host <POLICY_SERVER_IP> \
  --prompt "perform the manipulation task" \
  --publish
```

The model waypoints are stepped at 45 Hz by default. `main.py` converts each
absolute rot6d waypoint to a quaternion once and submits it to the ROS2
interface as the latest target. A separate timer publishes at 200 Hz by
default, linearly interpolating positions and gripper triggers and using
shortest-path quaternion nlerp for orientation. Every ramp starts from the
last command actually published, including when a new inference chunk begins;
the first ramp starts from the measured EE poses. The first command publishes
`/api/fsm/enable=1`, then the command callback repeats that heartbeat every
four published frames (about 50 Hz at the default 200 Hz interpolation rate).

```bash
# 45 Hz model targets -> 77 Hz interpolated command stream
python examples/teleavatar_v2_ee/main.py --remote-host <POLICY_SERVER_IP> \
  --prompt "perform the manipulation task" --publish \
  --control-frequency 45 --interp-frequency 77

# 45 Hz model targets -> 200 Hz interpolated command stream (the defaults)
python examples/teleavatar_v2_ee/main.py --remote-host <POLICY_SERVER_IP> \
  --prompt "perform the manipulation task" --publish \
  --control-frequency 45 --interp-frequency 200
```

Use `--no-interpolate` for a zero-order-hold baseline at the selected target
rate. `--control-frequency` sets how quickly model waypoints are consumed;
`--interp-frequency` only sets the timer publication rate.
`--open-loop-horizon` controls how many model waypoints are used before
observing and inferring again.
When `--rtc` is enabled, `--open-loop-horizon` is ignored because
`RTCActionBroker` returns one action per control tick and handles chunk overlap
on the server.

Before accepting every model target, the client compares its EE pose with the
latest measured current pose. It stops and disables output if the default
position/orientation limits (0.20 m / 0.80 rad) are exceeded. Adjust them with
`--max-position-error` and `--max-orientation-error`; use `0` to disable an
individual check.

## Observation and Command Semantics

The client sends the raw keys consumed by `TeleavatarEEInputs`:

- `observation/images/head_camera`: head left eye
- `observation/images/left_color`: left-wrist inner (right) eye
- `observation/images/right_color`: right-wrist inner (left) eye
- `observation/state`: converter-compatible 62D state with both current EE
  poses at indices 48:55 and 55:62
- `prompt`

The server converts both current poses to absolute position + row-wise
rotation-6D, predicts relative SE(3) waypoints for both arms, and converts the
result back to absolute pose9 before sending it to this client. The client
therefore must not compose either pose a second time. It converts each
rotation-6D waypoint to a quaternion once, interpolates in quaternion space,
and publishes that quaternion directly in the ROS `Pose` message.

The gripper API convention is `0=open, 1=closed`. Since this robot API does not
provide gripper position feedback, the observation tracks the last command
published by this process. `--initial-left-gripper-trigger` and
`--initial-right-gripper-trigger` default to `0.0`; set them to the actual
physical gripper states at startup.

On input loss, invalid/non-finite model output, Ctrl+C, or normal exit, the
client stops action playback. If real output was enabled, it publishes
`/api/fsm/enable=0` before shutting down.

## Inspect or Replay a Dataset Episode

Use the EE replay tool to verify the converted parquet data before starting a
policy. The explicit `--dry-run` mode reads one episode and prints both arm pose ranges,
quaternions, gripper trigger ranges, and the largest per-frame translation
step; it does not need ROS2 and publishes nothing:

```bash
python examples/teleavatar_v2_ee/replay_episode.py \
  --dataset <dataset_path> --episode 0 --dry-run
```

For a measured-trajectory check, select `--source state`. The default
`--source action` reads absolute EE targets from the recorded action column.
After checking the dry-run output, an optional real replay ramps from the
current left/right EE poses to frame 0 before publishing both API targets:

```bash
export ROS_DOMAIN_ID=29
python examples/teleavatar_v2_ee/replay_episode.py \
  --dataset <dataset_path> --episode 0 --source action --ramp-s 5
```

Do not run this alongside the policy client or another `/api/*` publisher.
