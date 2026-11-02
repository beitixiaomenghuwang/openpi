"""SDK-only bimanual EE observations and control; no ROS or RTP dependency."""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import logging
from pathlib import Path
import sys
import sysconfig
import threading
import time

import numpy as np
from openpi_client import image_tools

from examples.teleavatar_v2_ee.se3_trajectory import SE3Interpolator
from examples.teleavatar_v2_ee_sdk.sdk_video_interface import SDKH265VideoInterface

_CAMERA_VIEWS = {
    "head_camera": "head_left_eye",
    "left_color": "left_wrist_right_eye",
    "right_color": "right_wrist_left_eye",
}
# Periodic FeedbackFreshness.Source values. MACHINE_ERROR is not a heartbeat;
# check its actual fault status below, as in Teleavatar2/robot_zero_ee.py.
_REQUIRED_SOURCES = {"FSM": 0, "LEFT_ARM": 2, "RIGHT_ARM": 3, "LEFT_EE": 6, "RIGHT_EE": 7}
# TA2_dev/1.2.1 reserves the event send slot independently of FSM feedback.
_CONTROL_ACTION_SEND_DURATION = 0.150
# Full state()/video checks in the target thread; target updates run faster.
_HEALTH_CHECK_PERIOD = 0.010


@dataclasses.dataclass
class SDKConfig:
    robot_ip: str = ""
    """Robot SDK address (required)."""
    robot_port: int = 4433
    server_name: str = "quic_gateway"
    sdk_root: str = ""
    """Native build location; defaults to Teleavatar2 beside the OpenPI checkout."""
    connect_lib_path: str = ""
    ca_path: str = ""
    client_cert_path: str = ""
    client_key_path: str = ""
    control_frequency: float = 45.0
    """Policy waypoint rate; must match the training data cadence (gift_box_2_3_lerobot_45fps)."""
    interp_frequency: float = 500.0
    """Python target-update rate. The native loop samples the latest target at its own ~100 Hz,
    so a faster update keeps each sampled target fresher (less velocity jitter)."""
    sdk_command_rate_hz: float = 200.0
    """Native SDK command packet rate requested at connect; the server may cap it (~100 Hz observed)."""
    interpolate: bool = True
    sensor_timeout: float = 1.0
    initial_data_timeout: float = 30.0
    initial_action_timeout: float = 30.0
    """Maximum wait for the first policy target after READY, including RTC warmup."""
    action_timeout: float = 5.0
    """Stop output if the policy submits no new target for this many seconds."""
    image_size: int = 224
    """Letterbox camera views to this square size on the client with the same PIL resize_with_pad
    as the server's ResizeImages (identical pixels, ~10x smaller requests); 0 sends full crops."""
    video_decoder: str = "nvh265dec max-display-delay=0"
    video_layout_version: int | None = None
    """Optional expected SDK layout_version; dimensions must be 1280x2720."""
    initial_left_gripper_trigger: float = 0.0
    initial_right_gripper_trigger: float = 0.0
    """Initial command-state estimates, using the old client's trigger convention."""
    publish: bool = False
    """Enable SDK motion output. Default: observations and inference only."""

    def validate(self) -> None:
        if not self.robot_ip.strip():
            raise ValueError("robot_ip is required")
        if not 1 <= self.robot_port <= 65535:
            raise ValueError("robot_port must be in [1, 65535]")
        for name in (
            "control_frequency",
            "interp_frequency",
            "sdk_command_rate_hz",
            "sensor_timeout",
            "initial_data_timeout",
            "initial_action_timeout",
            "action_timeout",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("initial_left_gripper_trigger", "initial_right_gripper_trigger"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.image_size < 0:
            raise ValueError("image_size cannot be negative")


def _load_sdk(sdk_root: str = ""):
    try:
        return importlib.import_module("ta2_sdk")
    except ModuleNotFoundError as error:
        if error.name != "ta2_sdk":
            raise

    root = Path(sdk_root).expanduser().resolve() if sdk_root else Path(__file__).resolve().parents[3] / "Teleavatar2"
    filename = "ta2_sdk" + sysconfig.get_config_var("EXT_SUFFIX")
    # Match the interpreter ABI in the two existing builds, as in robot_zero_ee.
    for build in ("build-python310", "build-python"):
        extension = root / build / "python" / filename
        if extension.is_file():
            spec = importlib.util.spec_from_file_location("ta2_sdk", extension)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules["ta2_sdk"] = module
            return module
    raise ModuleNotFoundError(
        f"No ta2_sdk build matching {sys.executable} ({filename}) under {root}. "
        "Build with this interpreter; use --sdk-root or PYTHONPATH for another SDK location.",
        name="ta2_sdk",
    )


def _pose_array(pose: dict) -> np.ndarray:
    if "position" not in pose or "orientation" not in pose:
        raise ValueError("Missing position or orientation in EE feedback")
    position, orientation = pose["position"], pose["orientation"]
    value = np.array([position[axis] for axis in "xyz"] + [orientation[axis] for axis in "xyzw"], dtype=np.float64)
    norm = float(np.linalg.norm(value[3:]))
    if not np.all(np.isfinite(value)) or norm < 1e-7:
        raise ValueError("Invalid EE feedback pose")
    value[3:] /= norm
    return value


def _state_errors(snapshot: dict, timeout: float, *, controlling: bool = False) -> list[str]:
    errors = []
    if not snapshot["connected"] or not snapshot["session_active"]:
        errors.append("SDK connection/session unavailable")
    recv_age = snapshot["recv_age_s"]
    if recv_age is None or not np.isfinite(recv_age) or recv_age > timeout:
        errors.append(f"SDK state missing/stale (receive age={recv_age})")
    frame = snapshot["frame"]
    feedback = frame.get("feedback")
    if feedback is None:
        errors.append("SDK feedback freshness missing")
    else:
        source_timeout = min(timeout, feedback["stale_after_us"] / 1e6)
        for name, index in _REQUIRED_SOURCES.items():
            bit = 1 << index
            ages = feedback["source_age_us"]
            if not feedback["received_mask"] & bit or not feedback["fresh_mask"] & bit:
                errors.append(f"SDK {name} feedback missing/stale")
            elif index >= len(ages) or recv_age is None or ages[index] / 1e6 + recv_age > source_timeout:
                errors.append(f"SDK {name} feedback too old")
    for arm in ("left", "right"):
        try:
            _pose_array(frame.get("ee", {}).get(f"{arm}_current", {}))
        except (KeyError, ValueError) as error:
            errors.append(f"{arm} EE feedback: {error}")
    status = frame.get("status")
    if status is None:
        errors.append("SDK robot status missing")
    else:
        if status["fault_active"] or status["machine_error_code"] or status["fsm"] == "FSM_STATE_ERROR":
            arms = "; ".join(
                f"{arm} tracking_error={status.get(arm, {}).get('tracking_error', 'n/a')} "
                f"reject={status.get(arm, {}).get('last_reject', 'REJECT_REASON_NONE')}"
                for arm in ("left", "right")
            )
            errors.append(
                f"Robot fault: FSM={status['fsm']}, code={status['machine_error_code']}, "
                f"fault_active={status['fault_active']}, {arms}"
            )
        if controlling:
            if status["fsm"] != "FSM_STATE_READY" or not status["arm_authority_active"]:
                errors.append(f"Dual-arm control unavailable: FSM={status['fsm']}")
            for arm in ("left", "right"):
                reject = status.get(arm, {}).get("last_reject", "REJECT_REASON_NONE")
                if reject != "REJECT_REASON_NONE":
                    errors.append(f"{arm} command rejection: {reject}")
    return errors


class TeleavatarV2EESDKInterface:
    """Keep the old policy observation/action contract over the native SDK.

    Pose frames, units and TCP were confirmed unchanged. Gripper outputs are
    normalized force triggers, not widths or motor angles. The observation
    retains the old last-command estimate; raw motor radians are not normalized
    without calibration.
    """

    def __init__(
        self,
        config: SDKConfig,
        *,
        robot=None,
        video=None,
        camera_views: dict[str, str] | None = None,
        interpolate_gripper: bool = True,
    ) -> None:
        config.validate()
        self.config = config
        self._robot = robot
        self._video = video
        # Policy image key -> SDK view; only these views are sent and freshness-checked.
        self._camera_views = dict(_CAMERA_VIEWS if camera_views is None else camera_views)
        self._command_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_error: str | None = None
        self._motion_started = False
        self._output_enabled = False
        self._enabled_at: float | None = None
        self._target_at: float | None = None
        self._fault_event: str | None = None
        self._gripper_triggers = np.array(
            [config.initial_left_gripper_trigger, config.initial_right_gripper_trigger], dtype=np.float64
        )
        self._trajectory = SE3Interpolator(
            duration=1.0 / config.control_frequency, interpolate_gripper=interpolate_gripper
        )

    def connect(self) -> None:
        if self._robot is None:
            ta2_sdk = _load_sdk(self.config.sdk_root)
            self._robot = ta2_sdk.RobotClient()
            logging.info("TA2 native SDK version: %s (%s)", ta2_sdk.__version__, ta2_sdk.__file__)
        requested_rate = self.config.sdk_command_rate_hz if self.config.interpolate else self.config.control_frequency
        self._robot.connect(
            self.config.robot_ip,
            port=self.config.robot_port,
            server_name=self.config.server_name,
            connect_lib_path=self.config.connect_lib_path,
            ca_path=self.config.ca_path,
            client_cert_path=self.config.client_cert_path,
            client_key_path=self.config.client_key_path,
            desired_cmd_rate_hz=int(np.ceil(requested_rate)),
            enable_composite_video=True,
        )
        capabilities = self._robot.capabilities()
        if not capabilities["has_left_arm"] or not capabilities["has_right_arm"]:
            raise RuntimeError("The bimanual policy requires both SDK arms")
        logging.info("SDK capabilities: %s", capabilities)
        logging.info(
            "Policy targets at %.1f Hz, Python target updates at %.1f Hz; native SDK command rate "
            "requested %.1f Hz, server recommends %s Hz",
            self.config.control_frequency,
            self.config.interp_frequency if self.config.interpolate else self.config.control_frequency,
            requested_rate,
            capabilities["recommended_cmd_rate_hz"],
        )
        logging.info(
            "Gripper observation uses last-command estimates: left=%.3f right=%.3f; motor radians are not used",
            *self._gripper_triggers,
        )
        if self._video is None:
            self._video = SDKH265VideoInterface(
                self._robot,
                decoder=self.config.video_decoder,
                expected_layout_version=self.config.video_layout_version,
            )
        self._video.start()

    def _video_errors(self) -> list[str]:
        if self._video is None:
            return ["SDK video not started"]
        if self._video.stream_ended():
            return [f"SDK video stopped: {self._video.error}"]
        now = time.monotonic()
        timestamps = self._video.get_image_timestamps()
        errors = []
        for camera, view in self._camera_views.items():
            timestamp = timestamps.get(view)
            if timestamp is None or now - timestamp > self.config.sensor_timeout:
                errors.append(f"SDK video:{camera} missing/stale")
        return errors

    def _drain_events(self) -> None:
        """Log SDK events; FaultEvent.detail explains machine_error_code (older builds lack take_events)."""
        take_events = getattr(self._robot, "take_events", None)
        if take_events is None:
            return
        for event in take_events():
            fault = event.get("fault")
            if fault is not None:
                self._fault_event = (
                    f"code={fault.get('machine_error_code', 0)} cleared={fault.get('cleared', False)} "
                    f"detail={fault.get('detail', '')!r}"
                )
                logging.error("SDK fault event: %s (stamp_us=%s)", self._fault_event, event.get("stamp_us"))
            elif any(key in event for key in ("session_ended", "session_resumed", "config_changed")):
                logging.warning("SDK event: %s", event)

    def _snapshot_errors(self, snapshot: dict, *, controlling: bool = False) -> list[str]:
        self._drain_events()
        errors = _state_errors(snapshot, self.config.sensor_timeout, controlling=controlling or self._output_enabled)
        if self._fault_event is not None and any(error.startswith("Robot fault") for error in errors):
            errors.append(f"last SDK fault event: {self._fault_event}")
        errors.extend(self._video_errors())
        if self._worker_error is not None:
            errors.append(self._worker_error)
        return errors

    def sensor_errors(self) -> list[str]:
        if self._robot is None:
            return ["SDK not connected"]
        return self._snapshot_errors(self._robot.state())

    def wait_for_initial_data(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        next_log = 0.0
        while time.monotonic() < deadline:
            errors = self.sensor_errors()
            if not errors:
                return True
            if self._video is not None and self._video.stream_ended():
                raise RuntimeError("; ".join(errors))
            if time.monotonic() >= next_log:
                logging.info("Waiting for SDK inputs: %s", "; ".join(errors))
                next_log = time.monotonic() + 2.0
            time.sleep(0.05)
        logging.error("Timed out waiting for SDK inputs: %s", "; ".join(self.sensor_errors()))
        return False

    def get_policy_observation(self, prompt: str) -> dict:
        snapshot = self._robot.state()
        errors = self._snapshot_errors(snapshot)
        if errors:
            raise RuntimeError("Observation unavailable: " + "; ".join(errors))
        views = list(self._camera_views.values())
        images, timestamps = self._video.get_latest_images_with_timestamps(views)
        # Check the copied frames, not just the preceding video-status read.
        if any(time.monotonic() - timestamps[view] > self.config.sensor_timeout for view in views):
            raise RuntimeError("Copied SDK video is stale")
        with self._command_lock:
            triggers = self._gripper_triggers.copy()
        state = np.zeros(62, dtype=np.float32)
        for index, arm in enumerate(("left", "right")):
            state[48 + 7 * index : 55 + 7 * index] = _pose_array(snapshot["frame"]["ee"][f"{arm}_current"])
            state[7 + 8 * index] = 1.0 - triggers[index]
        return {
            "observation/state": state,
            "prompt": prompt,
            **{
                f"observation/images/{name}": self._model_image(images[view])
                for name, view in self._camera_views.items()
            },
        }

    def _model_image(self, image: np.ndarray) -> np.ndarray:
        size = self.config.image_size
        return image_tools.resize_with_pad(np.ascontiguousarray(image), size, size) if size else image

    def _transition_fsm(self, action: str, wanted: str, timeout: float) -> None:
        errors = self.sensor_errors()
        if errors:
            raise RuntimeError("SDK startup failed: " + "; ".join(errors))
        logging.info("Requesting dual-arm %s", action)
        self._robot.request_control_action(action)
        sent_at = time.monotonic()
        send_until = sent_at + _CONTROL_ACTION_SEND_DURATION
        deadline = sent_at + timeout
        status = {}
        previous_fsm = None
        while time.monotonic() < deadline:
            snapshot = self._robot.state()
            errors = self._snapshot_errors(snapshot)
            if errors:
                raise RuntimeError("SDK startup failed: " + "; ".join(errors))
            status = snapshot["frame"]["status"]
            if status["fsm"] != previous_fsm:
                logging.info("SDK FSM=%s; waiting for %s", status["fsm"], wanted)
                previous_fsm = status["fsm"]
            # A fast acknowledgement does not release the native send slot.
            # Continue feedback checks while it drains; never replay events.
            if status["fsm"] == wanted and time.monotonic() >= send_until:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"SDK FSM did not reach {wanted} within {timeout}s; "
            f"current={status.get('fsm')}, arm_authority_active={status.get('arm_authority_active')}, "
            f"left_reject={status.get('left', {}).get('last_reject', 'REJECT_REASON_NONE')}, "
            f"right_reject={status.get('right', {}).get('last_reject', 'REJECT_REASON_NONE')}, "
            f"SDK error={self._robot.last_error!r}"
        )

    def enable_output(self) -> None:
        if not self.config.publish:
            raise RuntimeError("Motion output requires --publish")
        snapshot = self._robot.state()
        errors = self._snapshot_errors(snapshot)
        if errors:
            raise RuntimeError("Cannot enable SDK output: " + "; ".join(errors))
        if self._output_enabled:
            return
        start = np.empty(16, dtype=np.float64)
        for index, arm in enumerate(("left", "right")):
            start[8 * index : 8 * index + 7] = _pose_array(snapshot["frame"]["ee"][f"{arm}_current"])
            start[8 * index + 7] = self._gripper_triggers[index]
        try:
            self._motion_started = True
            self._robot.set_pose_targets(start[:7], start[8:15])
            self._robot.set_enabled(True)
            self._robot.set_engage(True)
            self._robot.start_command_loop()
            time.sleep(0.3)  # Same admission sequence as the SDK dual-arm pose example.
            self._transition_fsm("START", "FSM_STATE_SLOW_START", 5.0)
            self._transition_fsm("CONFIRM", "FSM_STATE_READY", 10.0)
            snapshot = self._robot.state()
            errors = self._snapshot_errors(snapshot, controlling=True)
            if errors:
                raise RuntimeError("SDK admission failed: " + "; ".join(errors))
            # Seed interpolation from fresh measured poses after admission.
            for index, arm in enumerate(("left", "right")):
                start[8 * index : 8 * index + 7] = _pose_array(snapshot["frame"]["ee"][f"{arm}_current"])
            with self._command_lock:
                self._trajectory.reset(start)
                self._output_enabled = True
                self._enabled_at = time.monotonic()
                self._target_at = None
            self._stop.clear()
            self._thread = threading.Thread(target=self._command_loop, name="ta2-sdk-targets", daemon=True)
            self._thread.start()
            logging.info("SDK dual-arm READY confirmed; policy output enabled")
        except BaseException:
            self.disable_output()
            raise

    def publish_quaternion_action(self, action: np.ndarray) -> None:
        target = np.array(action, dtype=np.float64, copy=True)
        if target.shape != (16,) or not np.all(np.isfinite(target)):
            raise ValueError("Expected a finite 16D absolute bimanual quaternion action")
        for start in (0, 8):
            norm = float(np.linalg.norm(target[start + 3 : start + 7]))
            if norm < 1e-7:
                raise ValueError("Target quaternion has near-zero norm")
            target[start + 3 : start + 7] /= norm
            target[start + 7] = np.clip(target[start + 7], 0.0, 1.0)
        if not self._output_enabled:
            self.enable_output()
        errors = self.sensor_errors()
        if errors:
            self.disable_output()
            raise RuntimeError("Cannot submit SDK target: " + "; ".join(errors))
        try:
            with self._command_lock:
                if not self._output_enabled:
                    raise RuntimeError(self._worker_error or "SDK output disabled")
                self._target_at = time.monotonic()
                if self.config.interpolate:
                    self._trajectory.set_target(target)
                else:
                    self._send_command(target)
        except BaseException:
            self.disable_output()
            raise

    def _send_command(self, action: np.ndarray) -> None:
        self._robot.set_pose_targets(action[:7], action[8:15])
        self._robot.set_gripper_efforts(float(action[7]), float(action[15]))
        self._gripper_triggers[:] = action[[7, 15]]

    def _command_loop(self) -> None:
        period = 1.0 / (self.config.interp_frequency if self.config.interpolate else self.config.control_frequency)
        previous = time.monotonic()
        next_check = previous
        while not self._stop.wait(period):
            try:
                if time.monotonic() >= next_check:
                    errors = self.sensor_errors()
                    if errors:
                        raise RuntimeError("; ".join(errors))
                    next_check = time.monotonic() + _HEALTH_CHECK_PERIOD
                now = time.monotonic()
                with self._command_lock:
                    if not self._output_enabled:
                        return
                    if self._target_at is None and now - self._enabled_at > self.config.initial_action_timeout:
                        raise RuntimeError("First policy target timed out; stopping SDK command loop")
                    if self._target_at is not None:
                        if now - self._target_at > self.config.action_timeout:
                            raise RuntimeError("Policy target timed out; stopping SDK command loop")
                        if self.config.interpolate:
                            dt = min(now - previous, max(4.0 * period, 1.0 / self.config.control_frequency))
                            self._send_command(self._trajectory.step(max(dt, 1e-6)))
                previous = now
            except Exception as error:
                self._worker_error = f"SDK output stopped: {error}"
                logging.error("%s", self._worker_error)
                try:
                    self._stop_sdk_output()
                except Exception:
                    logging.exception("SDK output cleanup failed")
                return

    def _stop_sdk_output(self) -> None:
        with self._command_lock:
            self._output_enabled = False
            self._enabled_at = None
            self._target_at = None
            self._trajectory.reset()
            if not self._motion_started:
                return
            self._motion_started = False
            errors = []
            # Attempt every stop operation even if an earlier SDK call fails.
            for operation in (
                lambda: self._robot.set_enabled(False),
                self._robot.clear_grippers,
                lambda: self._robot.set_engage(False),
                # STOP preempts pending START/CONFIRM without a send-slot wait.
                lambda: self._robot.request_control_action("STOP"),
            ):
                try:
                    operation()
                except Exception as error:
                    errors.append(str(error))
            time.sleep(0.2)  # Allow the SDK command loop to transmit the stop event.
            try:
                self._robot.stop_command_loop()
            except Exception as error:
                errors.append(str(error))
            if errors:
                raise RuntimeError("SDK stop errors: " + "; ".join(errors))

    def disable_output(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("SDK target thread did not stop")
            self._thread = None
        self._stop_sdk_output()

    def close(self) -> None:
        try:
            self.disable_output()
        finally:
            try:
                if self._video is not None:
                    self._video.stop()
            finally:
                if self._robot is not None:
                    try:
                        # Fault details can arrive just after the state that stopped output.
                        self._drain_events()
                    finally:
                        self._robot.disconnect()
