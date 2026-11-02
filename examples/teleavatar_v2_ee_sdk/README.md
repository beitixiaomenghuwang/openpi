# TeleAvatar V2 EE deployment through the SDK

This client replaces ROS2 state/control and RTP video with the TA2 C++ SDK through
the `ta2_sdk` Python extension. The original `teleavatar_v2_ee` files are unchanged.
The client reuses their ROS-independent `SE3Interpolator`.

The policy server, checkpoint, normalization statistics, 20D bimanual action
representation, and optional RTC protocol stay the same. The robot pose frame,
units, and TCP were confirmed unchanged: each arm uses its own shoulder frame.

## Dependencies and native extension

Build the extension from `Teleavatar2/python` as described in that directory's
README. Use the same Python interpreter for the extension and this script.
The client also needs `numpy`, `tyro`, `openpi_client`, PyGObject (`gi`), and
GStreamer Gst/GstVideo with `appsrc`, `h265parse`, an H.265 decoder, `videoconvert`,
and `appsink`. The pipeline uses GStreamer 1.20 appsrc queue controls.

```bash
export TA2_SDK_ROOT=/path/to/Teleavatar2
export PYTHONPATH="$TA2_SDK_ROOT/build-python/python${PYTHONPATH:+:$PYTHONPATH}"
cd /path/to/openpi
python -c 'import ta2_sdk; import gi; print(ta2_sdk.__version__)'
python examples/teleavatar_v2_ee_sdk/main.py --help
```

On this machine, the existing `ros2_rtc` environment has Python 3.10, `gi`, and
`openpi_client`. Its matching extension is in `Teleavatar2/build-python310/python`:

```bash
conda activate ros2_rtc
cd /home/caslx/Robotics/openpi
```

Like the corrected `Teleavatar2/robot_zero_ee.py`, the client first imports an
installed/on-`PYTHONPATH` `ta2_sdk`. Otherwise it looks in the sibling `Teleavatar2`
directory's `build-python310/python` and `build-python/python`, matching the
current interpreter's extension suffix. No `PYTHONPATH` export is needed for
this checkout layout. Use `--sdk-root /path/to/Teleavatar2` for a different SDK
location, or `PYTHONPATH` for a differently named build directory. A missing
matching build or native dependency is an error, not a switch to another Python.

The environment's name does not imply a ROS dependency: this client does not
import `rclpy`, ROS messages, or the original RTP module. Do not add Python 3.10
system `gi` binaries to a Python 3.11 path; use matching bindings/interpreters.

## Run

Start the policy server on the inference machine using the checkpoint's original
configuration. This example assumes a `pi05_teleavatar_v2_ee` checkpoint; replace
both the configuration and checkpoint path with the ones actually trained:

```bash
cd /path/to/openpi
.venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_teleavatar_v2_ee \
  --policy.dir /path/to/trained/checkpoint
```

`pi05_teleavatar_v2_ee` uses `discrete_state_input=False`, matching the
`pi05_gift_box_3` training run: the pi0.5 model receives no state. The server
still needs the EE poses in `observation/state` to restore relative actions.
The checkpoint stores no training config, so serving it with a config that sets
`discrete_state_input=True` would silently add state tokens to the prompt.

`--control-frequency` defaults to 45 Hz, the cadence of the
`gift_box_2_3_lerobot_45fps` training data; change it only for a checkpoint
trained at another cadence. First run without motion output, from the
SDK/PyGObject environment above:

```bash
python examples/teleavatar_v2_ee_sdk/main.py \
  --robot-ip "<ROBOT_IP>" \
  --remote-host "<POLICY_SERVER_IP>" \
  --prompt "perform the manipulation task" \
  --max-steps 90
```

This opens an SDK control session, reads state/video, and runs inference, but
does not engage, start a command loop, or send pose/gripper/FSM commands. It is
not a passive subscription independent of other clients' robot sessions.

After checking the camera views and robot feedback, add `--publish` for control:

```bash
python examples/teleavatar_v2_ee_sdk/main.py \
  --robot-ip "<ROBOT_IP>" \
  --remote-host "<POLICY_SERVER_IP>" \
  --prompt "perform the manipulation task" \
  --publish
```

Connection defaults are port `4433` and TLS `server_name=quic_gateway`. Use
`--robot-port`, `--server-name`, `--connect-lib-path`, `--ca-path`,
`--client-cert-path`, and `--client-key-path` when the deployment requires explicit
paths/settings. Keep credentials in the SDK package/configuration, not source code.
The robot-side SDK gateway must already be running. No `ROS_DOMAIN_ID` is used;
the robot link is SDK QUIC/TLS, while policy inference uses WebSocket port `8000`.
Run the zeroing script separately if needed and let it exit before starting this
client. The client does not run zeroing automatically. The C++ examples are
standalone clients; none needs to run alongside OpenPI.

The default decoder is `nvh265dec max-display-delay=0`. If the environment uses
a software decoder, select it explicitly:

```bash
python examples/teleavatar_v2_ee_sdk/main.py \
  --robot-ip "<ROBOT_IP>" --remote-host "<POLICY_SERVER_IP>" \
  --video-decoder libde265dec --max-steps 30
```

For RTC, add `--rtc` to the client and enable it on the server. For an ordinary
EE checkpoint using guided RTC, the server command is:

```bash
.venv/bin/python scripts/serve_policy.py \
  --port 8000 --rtc.enabled --rtc.mode GUIDED \
  policy:checkpoint \
  --policy.config pi05_teleavatar_v2_ee \
  --policy.dir /path/to/trained/checkpoint
```

The client retains the old warmup, calibration, inference-delay, and
execution-horizon options. Measure latency again after changing the transport.

## Record inference images and action chunks

Enable recording on the policy server to inspect the camera inputs and gripper
outputs from the same inference request. Keep `--record` and `--record-dir`
before `policy:checkpoint`; the SDK client command stays the same:

```bash
uv run scripts/serve_policy.py \
  --port 8000 --record --record-dir /path/to/records \
  policy:checkpoint \
  --policy.config pi05_teleavatar_v2_ee \
  --policy.dir /path/to/trained/checkpoint
```

Each server run creates a UTC timestamped session directory under `--record-dir`
(default `policy_records`) and logs its location. Each successful inference saves:

- `step_N.npy`: a flattened dictionary containing the original request under
  `inputs/`, the complete policy output under `outputs/`, and transformed inputs under
  `model_inputs/`. The latter are captured after the actual input transforms,
  including image crop/resize, before batching and image normalization to `[-1,1]`.
  `recording/step`, `recording/infer_start_time_ns`, and
  `recording/infer_end_time_ns` identify the request and its wall-clock timing.
- `step_N/actions.npy` and `step_N/actions.csv`: the full returned action chunk.
  CSV columns are named `action_0`, `action_1`, etc.
- `step_N/images/inputs__observation__images__head_camera.png`,
  `inputs__observation__images__left_color.png`, and
  `inputs__observation__images__right_color.png`: the RGB images received from
  the SDK client, normally 224x224 with its default `--image-size 224`.
- `step_N/images/model_inputs__image__base_0_rgb.png`,
  `model_inputs__image__left_wrist_0_rgb.png`, and
  `model_inputs__image__right_wrist_0_rgb.png`: the corresponding images after
  the server's input transforms.

PNG export preserves uint8 HWC RGB arrays without further resizing or channel
conversion. Arrays in other formats remain in `step_N.npy` without a PNG preview.
The original request also retains `observation/state`, `prompt`, and any RTC
fields. To inspect a saved request and its left/right gripper outputs:

```python
from pathlib import Path

import numpy as np

session = Path("/path/to/records/<UTC_SESSION>")
step = 0
record = np.load(session / f"step_{step}.npy", allow_pickle=True).item()
actions = np.load(session / f"step_{step}" / "actions.npy")
print(record["inputs/prompt"])
print(record["inputs/observation/images/head_camera"].shape)
print(actions.shape)
print(actions[:, [9, 19]])  # Left/right triggers: 0 opens, 1 closes.
```

For TeleAvatar EE, the saved chunk has shape `(N, 20)` and already contains
absolute EE poses. It includes every returned waypoint, before the client applies
`--open-loop-horizon`, `--max-steps`, quaternion conversion, or chunk blending;
it does not say which waypoints were executed. RTC warmup and calibration
inferences are recorded too. Recording writes to disk synchronously and adds
inference latency; include that overhead when measuring RTC timing.

## State, actions, and grippers

The policy observation retains these keys:

- `observation/state`: float32 length 62; left EE at `48:55`, right EE at `55:62`.
- `observation/images/left_color`: left wrist's right eye.
- `observation/images/right_color`: right wrist's left eye.
- `observation/images/head_camera`: head's left eye (always sent; TeleAvatar EE policies always use it).
- `prompt`.

`TeleavatarV2EESDKInterface(camera_views=..., interpolate_gripper=...)` defaults
to the three views above and linear trigger interpolation. Other clients, such as
`examples/yubi_umi_sdk`, use these options to send and freshness-check only their
own views, and to pass each trigger through unchanged.

Server outputs are already **absolute** poses: each arm contributes position (3),
row-wise rotation-6D (6), and gripper trigger (1). The client changes rotation-6D
to an `xyzw` quaternion and does not compose the current pose a second time.

Grippers use the current SDK's normalized force command, as confirmed for this
migration. Values are clipped to `[0,1]`: `0` drives opening, `0.10` is the
zero-force point, and `1` drives closing. This is not a desired geometric width.
The SDK's motor angle in radians is feedback only; no angle-target conversion
or position controller is invented here.

The raw state entries `7` and `15` hold the last-command estimate `1 - trigger`.
With `discrete_state_input=False`, the model does not read them; they only keep
the 62D layout intact. Startup values come from `--initial-left-gripper-trigger`
and `--initial-right-gripper-trigger` (both default `0`).

### Test intermediate gripper commands

`gripper_sweep.py` tests the native force commands without a policy server or
camera stream. Use the same Python environment as this client (`ros2_rtc` on this
machine). It reuses the client's SDK loader and feedback checks. Without
`--publish`, it only prints the test configuration and does not connect.

Hold each right-gripper command for two seconds:

```bash
python examples/teleavatar_v2_ee_sdk/gripper_sweep.py \
  --robot-ip "<ROBOT_IP>" --side right \
  --values 0 0.05 0.10 0.15 0.20 0.30 0.50 0.70 1.0 0 --hold 2 --publish
```

For a continuous `0 -> 1 -> 0` scan, hold `0` for five seconds, then ramp to `1`
over five seconds and back to `0` over five seconds:

```bash
python examples/teleavatar_v2_ee_sdk/gripper_sweep.py \
  --robot-ip "<ROBOT_IP>" --side right \
  --values 0 1 0 --ramp --hold 5 --hz 20 --publish
```

Use `--side left` or `--side both` to change the tested grippers. The Python
binding always updates both grippers: an unselected gripper receives `0.10`,
which may release an existing grasp. Both arms are enabled with their measured
initial EE poses as hold targets, using the same START/CONFIRM admission as the
client. This is active position holding; physical arm motion cannot be ruled out.
Run with empty grippers and stop other robot control clients first.

The log prints the commands and the latest available motor-angle (`rad`) and
torque (`Nm`) feedback, marking missing/stale gripper feedback explicitly.
Feedback is asynchronous and is not a geometric opening-width measurement.
Normal completion sends `0.10` to both grippers before shutdown. Completion,
Ctrl+C, and errors disable the arms, clear gripper intents, request STOP, stop
the native command loop, and disconnect; clearing an intent alone does not
guarantee immediate force removal.

The `0=open, 0.10=neutral, 1=close` convention follows this deployment and SDK
`03_cartesian_gripper.cpp`. SDK `08_gripper_joint.cpp` has a conflicting comment
calling `0.04` a gentle close. The script does not infer an opening width or a
linear force mapping from the command; observe the actual robot response.

## Startup, cadence, and stopping

After metadata and sensor checks, motion mode sets both measured poses as hold
targets, enables both arms, engages, and starts the native command loop. It requests
`START`, waits for `SLOW_START`, requests `CONFIRM`, and waits for `READY`.
Each transition also waits for the SDK's 150 ms event-send window to finish;
early FSM feedback does not release that slot. Feedback/video checks continue
during the wait and accepted events are not replayed. After READY, authority and
rejection status are checked, and interpolation starts from fresh measured poses.
Only then does it acquire the first inference observation, so an old observation
is not replayed across the potentially slow FSM startup. No gripper intent is
sent until the first valid policy action.

`--control-frequency` defaults to 45 Hz. Policy targets follow fixed deadlines,
so per-step overhead does not stretch every period; a step that is a full period
late restarts the cadence instead of bursting. `--interp-frequency` defaults to
500 Hz and controls the Python target-update thread. The native SDK loop samples
the latest target at its own rate (0x32 measured ~100 Hz), so faster updates keep
each sampled target fresher. The full state/video health check in that thread
runs every 10 ms. `--no-interpolate` submits policy waypoints directly.
`--sdk-command-rate-hz` (default 200) is the native packet rate requested at
connect; the server may cap it. Python scheduling is not hard realtime. The
reused interpolator imposes no velocity, acceleration, or jerk limits; existing
robot-side limits remain active.

Chunk transitions are blended. A new chunk is anchored on the measured pose, which
lags the previous chunk's last target while the arm moves; the 2026-10-23 log
showed right-arm snaps of 15–37 mm / up to 20 deg that matched this lag. Without
RTC, `--chunk-blend-steps` (default 10 targets, 0.22 s at 45 Hz) starts each new
chunk at the previous last target and fades the position/rotation offset out with
a cosine weight; grippers are not blended. With RTC, `--rtc-blend-steps` (default
5) cross-fades the replaced and new chunks' actions for the same control steps.

With RTC, the broker receives an observation supplier and builds observations
only when it launches an inference request, on its worker thread; playback ticks
only check input freshness and submit the next action. The logged `broker wait`
measures queue waiting, not GPU inference. `--image-size` (default 224)
letterboxes the requested camera views on the client with the same PIL
`resize_with_pad` the server's `ResizeImages(224, 224)` uses, which then leaves
them unchanged: identical model pixels, about 0.45 MB instead of 4.3 MB per
request. `--image-size 0` sends full-resolution crops.

Robot-fault errors include `fault_active`, per-arm `tracking_error`
and rejection, and the latest SDK `FaultEvent` (code, detail, cleared) when the
binding provides `take_events()`; late fault events are logged on close.

Without RTC, the client takes one observation, waits for server inference, and
plays the first `--open-loop-horizon` actions (default 30) before observing again.
At 45 Hz, 30 actions span about 0.67 s, plus the blocking inference gap; the last
target remains held while waiting, so both arms pause together once per chunk.
Use RTC to overlap inference with execution. With RTC, startup performs two warmup and five
latency-calibration inferences by default, then background inference overlaps
with one action per control tick. `--open-loop-horizon` only applies without RTC.
The server preprocesses/normalizes observations and restores model predictions
to absolute EE actions; the client handles quaternion conversion and SDK output.

The output thread checks connection/session, EE and arm freshness, FSM, faults,
authority, command rejection, and image freshness even while inference is blocked.
Both SDK source age and local packet age count toward freshness. It also stops
after `--initial-action-timeout` (default 30 seconds) without a first target,
or `--action-timeout` (default 5 seconds) without a subsequent target. Set those
timeouts to match measured inference/RTC startup latency rather than silently
allowing indefinite command repetition. A worker fault requires restarting the
client; a later inference result does not automatically re-enable motion.
`MACHINE_ERROR` is not required to update as a periodic heartbeat; actual
`fault_active`, `machine_error_code`, and error FSM still block output. Missing
optional arm diagnostic submessages use the SDK's `REJECT_REASON_NONE` default;
an explicitly reported rejection still stops control.

Ctrl+C, normal exit, and errors disable arm intents/engage, request `STOP`, allow
the running SDK loop to send the stop event, stop that loop, stop video, and
disconnect. A stop request itself is not proof of physical stopping after a
network failure; the robot's own watchdog and emergency-stop behavior still apply.
`STOP` preempts a pending startup event immediately; it does not wait for the
START/CONFIRM send window.

## SDK video layout

The input is the SDK's 1280x2720 all-I Annex-B H.265 composite, not RTP packets.
Connection enables the SDK's `0x24` composite subscription. The C++ callback
copies the latest encoded frame; `take_video_frame()` passes owned bytes to the
Python decoding thread. Its pipeline is
`appsrc → h265parse → nvh265dec → videoconvert (RGB) → appsink`.
The client crops the decoded frame and sends RGB image arrays with state/prompt
to the policy server over WebSocket; the policy server does not receive H.265.
No RTP receiver, port 8890, `14_video` process, or intermediate video file is used.
Image age includes binding queue and decoder time, using the local
monotonic receive clock. Remote capture-to-receive latency needs separate clock
alignment and is not inferred from that age.

The six crop rectangles are preserved from the original RTP layout:

| View | x range | y range |
| --- | --- | --- |
| Head right | 160:1120 | 0:960 |
| Head left | 160:1120 | 960:1920 |
| Right wrist left | 0:640 | 1920:2320 |
| Right wrist right | 640:1280 | 1920:2320 |
| Left wrist left | 0:640 | 2320:2720 |
| Left wrist right | 640:1280 | 2320:2720 |

The SDK header documents composite dimensions but not the rectangle order or
numerical layout-version meanings. **Actual view placement remains to be checked
on the robot.** A different dimension or a runtime layout-version change fails
explicitly. Optionally use `--video-layout-version <N>` to enforce a verified
deployment version; no version number is guessed by default.

For a read-only image check, use the same Python environment and set the robot IP:

```python
from pathlib import Path

from PIL import Image
from examples.teleavatar_v2_ee_sdk.sdk_interface import SDKConfig, TeleavatarV2EESDKInterface

# image_size=0 keeps the full-resolution crops for checking the layout.
interface = TeleavatarV2EESDKInterface(SDKConfig(robot_ip="<ROBOT_IP>", image_size=0))
try:
    interface.connect()
    if not interface.wait_for_initial_data():
        raise RuntimeError("SDK inputs unavailable")
    observation = interface.get_policy_observation("inspect camera layout")
    output = Path("sdk-camera-check")
    output.mkdir(exist_ok=True)
    for name in ("head_camera", "left_color", "right_color"):
        rgb = observation[f"observation/images/{name}"]
        Image.fromarray(rgb).save(output / f"{name}.png")
finally:
    interface.close()
```

## Offline validation

```bash
# Run from the OpenPI root with the repository's test environment.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  examples/teleavatar_v2_ee_sdk examples/teleavatar_v2_ee/test_se3_trajectory.py -q
.venv/bin/ruff check examples/teleavatar_v2_ee_sdk
```

The local ROS pytest auto-plugin requires a missing `lark` package; disabling
unrelated automatic plugins leaves these tests unchanged. Run the video tests
with a Python that has `gi` to exercise real synthetic H.265 encode/decode:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest \
  examples/teleavatar_v2_ee_sdk/test_sdk_video_interface.py -q
```

Tests cover action representation, read-only behavior, state freshness, startup
and stopping using a fake transport, and real GStreamer video processing with
synthetic frames. No robot connection, real FSM admission, motion, network timing,
or policy success is claimed by these offline checks.
