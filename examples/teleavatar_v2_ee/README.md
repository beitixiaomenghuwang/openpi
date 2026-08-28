# TeleAvatar V2 Bimanual End-Effector Deployment

This example deploys an OpenPI end-effector policy directly on both TeleAvatar
V2 arms. It is independent from the joint-space `examples/teleavatar_v2`
example and from HIL: it does not import HIL code or publish `/hil/*` topics.

The matching training configs are:

- `pi0_teleavatar_v2_ee`
- `pi05_teleavatar_v2_ee`
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

For real control, add `--publish`. The client waits for Enter immediately
before it starts publishing:

```bash
python examples/teleavatar_v2_ee/main.py \
  --remote-host <POLICY_SERVER_IP> \
  --prompt "perform the manipulation task" \
  --publish
```

Defaults are 45 Hz and 30 actions per inference, matching the converter's
current 45 FPS dataset and the EE model's action horizon. Use
`--open-loop-horizon` to execute fewer actions before observing and inferring
again.

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
therefore must not compose either pose a second time; it only converts each
rotation-6D to a ROS quaternion.

The gripper API convention is `0=open, 1=closed`. Since this robot API does not
provide gripper position feedback, the observation tracks the last command
published by this process. `--initial-left-gripper-trigger` and
`--initial-right-gripper-trigger` default to `0.0`; set them to the actual
physical gripper states at startup.

On input loss, invalid/non-finite model output, Ctrl+C, or normal exit, the
client stops action playback. If real output was enabled, it publishes
`/api/fsm/enable=0` before shutting down.
