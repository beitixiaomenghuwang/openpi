"""Decode the SDK's owned Annex-B H.265 composite frames into RGB camera views."""

from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np

LOGGER = logging.getLogger(__name__)

# Pixel equivalents of teleavatar_v2/rtp_video_interface.py's six-view layout.
# Confirm the SDK stream uses this layout before enabling robot motion.
SPLIT_REGIONS = (
    ("head_right_eye", (160, 0, 1120, 960)),
    ("head_left_eye", (160, 960, 1120, 1920)),
    ("right_wrist_left_eye", (0, 1920, 640, 2320)),
    ("right_wrist_right_eye", (640, 1920, 1280, 2320)),
    ("left_wrist_left_eye", (0, 2320, 640, 2720)),
    ("left_wrist_right_eye", (640, 2320, 1280, 2720)),
)


def _split_frame(frame: np.ndarray) -> dict[str, np.ndarray]:
    if frame.shape != (2720, 1280, 3) or frame.dtype != np.uint8:
        raise ValueError(f"Expected 1280x2720 RGB uint8 composite, got {frame.shape} {frame.dtype}")
    return {name: frame[y1:y2, x1:x2].copy() for name, (x1, y1, x2, y2) in SPLIT_REGIONS}


def _rgb_from_buffer(data, width: int, height: int, stride: int, offset: int = 0) -> np.ndarray:
    """Respect GStreamer row padding and plane offset, and own the returned pixels."""
    if width <= 0 or height <= 0 or stride < width * 3 or offset < 0:
        raise ValueError("Invalid RGB dimensions, row stride, or plane offset")
    if len(data) < offset + (height - 1) * stride + width * 3:
        raise ValueError("Decoded RGB buffer is shorter than its dimensions and stride")
    return np.ndarray((height, width, 3), dtype=np.uint8, buffer=data, offset=offset, strides=(stride, 3, 1)).copy()


def _sample_to_rgb(sample, gst, gst_video) -> np.ndarray:
    caps = sample.get_caps()
    if caps is None or caps.get_size() == 0:
        raise RuntimeError("Decoded sample has no caps")
    if caps.get_structure(0).get_value("format") != "RGB":
        raise RuntimeError("Decoded sample is not RGB")
    info = gst_video.VideoInfo.new_from_caps(caps)
    buffer = sample.get_buffer()
    if buffer is None:
        raise RuntimeError("Decoded sample has no buffer")
    meta = gst_video.buffer_get_video_meta(buffer)
    stride = meta.stride[0] if meta is not None else info.stride[0]
    offset = meta.offset[0] if meta is not None else info.offset[0]
    ok, mapped = buffer.map(gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("Failed to map decoded RGB buffer")
    try:
        return _rgb_from_buffer(mapped.data, info.width, info.height, stride, offset)
    finally:
        buffer.unmap(mapped)


class SDKH265VideoInterface:
    """Decode outside the SDK callback; retain only the latest composite/views.

    Image timestamps use the local monotonic clock at SDK reception, including
    time spent waiting in the binding and decoder. Remote capture-clock offset
    is unknown, so these timestamps do not measure camera/network latency.
    """

    def __init__(
        self,
        robot,
        *,
        decoder: str = "nvh265dec max-display-delay=0",
        expected_layout_version: int | None = None,
        camera_name: str = "head_camera",
    ):
        self.robot = robot
        self.decoder = decoder
        self.expected_layout_version = expected_layout_version
        self.camera_name = camera_name
        self.error: str | None = None
        self._lock = threading.Lock()
        self._images: dict[str, np.ndarray] = {}
        self._image_timestamps: dict[str, float] = {}
        self._source_timestamps: dict[int, float] = {}
        self._layout_version: int | None = None
        self._stop = threading.Event()
        self._ended = threading.Event()
        self._ready_or_ended = threading.Event()
        self._thread: threading.Thread | None = None
        self._pipeline = None
        self._gst = None
        self._gst_video = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            import gi

            gi.require_version("Gst", "1.0")
            gi.require_version("GstVideo", "1.0")
            from gi.repository import Gst
            from gi.repository import GstVideo
        except (ImportError, ValueError) as exc:
            raise RuntimeError(
                "SDK video requires PyGObject and GStreamer Gst/GstVideo in this Python environment"
            ) from exc

        Gst.init(None)
        self._gst, self._gst_video = Gst, GstVideo
        # The SDK documents all-I frames, so dropping old encoded frames is safe.
        self._pipeline = Gst.parse_launch(
            "appsrc name=encoded_frames is-live=true format=time block=false "
            "max-buffers=1 max-bytes=0 max-time=0 leaky-type=downstream "
            'caps="video/x-h265,stream-format=byte-stream,alignment=au" '
            "! h265parse config-interval=-1 "
            f"! {self.decoder} ! videoconvert ! video/x-raw,format=RGB "
            "! appsink name=decoded_frames emit-signals=true max-buffers=1 drop=true sync=false"
        )
        self._pipeline.get_by_name("decoded_frames").connect("new-sample", self._on_sample)
        self._stop.clear()
        self._ended.clear()
        self._ready_or_ended.clear()
        self.error = None
        self._layout_version = None
        with self._lock:
            self._images.clear()
            self._image_timestamps.clear()
            self._source_timestamps.clear()
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Failed to start SDK H.265 decoder")
        self._thread = threading.Thread(target=self._run, name="sdk-h265-video", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("SDK H.265 decoder did not stop within 2 seconds")
        self._pipeline = None

    def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
        self._ready_or_ended.wait(timeout)
        with self._lock:
            return bool(self._images)

    def get_latest_images_with_timestamps(
        self, names: list[str] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Copy the latest views (all, or only ``names``) and their receive timestamps."""
        with self._lock:
            selected = self._images if names is None else {n: self._images[n] for n in names if n in self._images}
            return {name: frame.copy() for name, frame in selected.items()}, dict(self._image_timestamps)

    def get_image_timestamps(self) -> dict[str, float]:
        with self._lock:
            return dict(self._image_timestamps)

    def stream_ended(self) -> bool:
        return self._ended.is_set()

    def _validate_frame(self, frame: dict) -> float:
        if (frame["width"], frame["height"]) != (1280, 2720):
            raise ValueError(f"SDK video must be 1280x2720, got {frame['width']}x{frame['height']}")
        version = frame["layout_version"]
        expected = self.expected_layout_version if self.expected_layout_version is not None else self._layout_version
        if expected is not None and version != expected:
            raise ValueError(f"SDK video layout_version changed or mismatched: expected {expected}, got {version}")
        if self._layout_version is None:
            LOGGER.info("SDK video: 1280x2720 layout_version=%s; using the existing six-view crop layout", version)
            self._layout_version = version
        age = float(frame["recv_age_s"])
        if not math.isfinite(age) or age < 0:
            raise ValueError("SDK video recv_age_s must be finite and nonnegative")
        if not frame["data"]:
            raise ValueError("SDK video frame is empty")
        return time.monotonic() - age

    def _run(self) -> None:
        gst = self._gst
        pipeline = self._pipeline
        appsrc = pipeline.get_by_name("encoded_frames")
        bus = pipeline.get_bus()
        started_ns = time.monotonic_ns()
        try:
            while not self._stop.is_set():
                message = bus.timed_pop_filtered(0, gst.MessageType.ERROR | gst.MessageType.EOS)
                if message is not None:
                    if message.type == gst.MessageType.ERROR:
                        error, debug = message.parse_error()
                        raise RuntimeError(f"GStreamer: {error}; {debug}")
                    break
                frame = self.robot.take_video_frame(timeout_ms=100)
                if frame is None:
                    continue
                timestamp = self._validate_frame(frame)
                buffer = gst.Buffer.new_allocate(None, len(frame["data"]), None)
                buffer.fill(0, frame["data"])
                # Unique local PTS pairs decoded frames with their original receive times.
                buffer.pts = time.monotonic_ns() - started_ns
                with self._lock:
                    self._source_timestamps[buffer.pts] = timestamp
                    while len(self._source_timestamps) > 64:
                        del self._source_timestamps[next(iter(self._source_timestamps))]
                result = appsrc.emit("push-buffer", buffer)
                if result != gst.FlowReturn.OK and not self._stop.is_set():
                    raise RuntimeError(f"H.265 appsrc rejected a frame: {result}")
        except Exception as exc:
            self._fail(exc)
        finally:
            pipeline.set_state(gst.State.NULL)
            self._ended.set()
            self._ready_or_ended.set()

    def _on_sample(self, sink):
        gst = self._gst
        try:
            sample = sink.emit("pull-sample")
            if sample is None:
                raise RuntimeError("H.265 decoder returned no sample")
            pts = sample.get_buffer().pts
            if pts == gst.CLOCK_TIME_NONE:
                raise RuntimeError("H.265 decoder discarded the frame timestamp")
            with self._lock:
                timestamp = self._source_timestamps.pop(pts, None)
            if timestamp is None:
                # A delayed output whose metadata was evicted is not a fresh observation.
                return gst.FlowReturn.OK
            frame = _sample_to_rgb(sample, gst, self._gst_video)
            images = {self.camera_name: frame, **_split_frame(frame)}
            with self._lock:
                self._images = images
                self._image_timestamps = dict.fromkeys(images, timestamp)
            self._ready_or_ended.set()
            return gst.FlowReturn.OK
        except Exception as exc:
            self._fail(exc)
            return gst.FlowReturn.ERROR

    def _fail(self, error: Exception) -> None:
        self.error = str(error)
        LOGGER.error("SDK video failed: %s", error)
        self._stop.set()
        self._ended.set()
        self._ready_or_ended.set()
