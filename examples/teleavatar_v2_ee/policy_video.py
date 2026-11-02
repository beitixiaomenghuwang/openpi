"""Keep decoding callbacks short; prepare shared model images on request."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import threading

import numpy as np
from openpi_client import image_tools

# Boxes in the original 1280x2720 composite, matching TELEAVATAR_SPLIT_REGIONS.
VIEWS = {
    "head_left_eye": (160, 960, 1120, 1920),
    "left_wrist_right_eye": (640, 2320, 1280, 2720),
    "right_wrist_left_eye": (0, 1920, 640, 2320),
    "left_wrist_left_eye": (0, 2320, 640, 2720),
}
DEFAULT_VIEWS = ("head_left_eye", "left_wrist_right_eye", "right_wrist_left_eye")


def prepare_images(frame: np.ndarray, views: tuple[str, ...] = DEFAULT_VIEWS) -> np.ndarray:
    if frame.shape != (2720, 1280, 3) or frame.dtype != np.uint8:
        raise ValueError(f"Expected 1280x2720 RGB uint8 composite, got {frame.shape}, {frame.dtype}")
    # Pack crops before PIL conversion, avoiding its slow strided-array path.
    return np.stack(
        [
            image_tools.resize_with_pad(np.ascontiguousarray(frame[y1:y2, x1:x2]), 224, 224)
            for x1, y1, x2, y2 in (VIEWS[name] for name in views)
        ]
    )


def _decode_worker(
    buffer, timestamp, prepared_at, lock, request, ready, stop, error, parent_pid, port, payload, decoder, views
):
    # Import GI/GStreamer only in the video process, after spawn.
    video = None
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    images = np.frombuffer(buffer, dtype=np.uint8).reshape(len(views), 224, 224, 3)
    latest = None

    def publish(frame, received_at):
        nonlocal latest
        # _sample_to_rgb supplies an owned array. Retain it without copying or
        # resizing; the streaming thread must return to appsink immediately.
        latest = (frame, received_at)
        timestamp.value = received_at

    try:
        from examples.teleavatar_v2.rtp_video_interface import RTPH265VideoInterface

        video = RTPH265VideoInterface(
            port=port,
            payload=payload,
            decoder=decoder,
            frame_callback=publish,
            packet_stats=False,
        )
        video.start()
        while not stop.is_set() and os.getppid() == parent_pid:
            if video.stream_ended():
                raise RuntimeError("RTP decoder stopped")
            if not request.wait(0.1):
                continue
            if stop.is_set():
                break
            request.clear()
            if latest is None:
                raise RuntimeError("Policy video has no frame")
            frame, received_at = latest
            prepared = prepare_images(frame, views)
            if not lock.acquire(timeout=0.05):
                raise RuntimeError("Policy video shared-frame lock timed out")
            try:
                images[:] = prepared
                prepared_at.value = received_at
            finally:
                lock.release()
            ready.set()
    except Exception as exc:
        error.value = f"{type(exc).__name__}: {exc}".encode()[:1023]
        logging.exception("Policy video worker failed")
    finally:
        ready.set()  # wake a pending observation on failure/shutdown
        if video is not None:
            video.stop()


class PolicyVideo:
    """One bounded shared frame, with a short lock to avoid partially copied images."""

    def __init__(self, *, port: int, payload: int, decoder: str, views: tuple[str, ...] = DEFAULT_VIEWS):
        if not views or len(set(views)) != len(views) or set(views) - VIEWS.keys():
            raise ValueError(f"Expected distinct supported camera views, got {views}")
        self._views = tuple(views)
        image_shape = (len(self._views), 224, 224, 3)
        ctx = mp.get_context("spawn")
        self._buffer = ctx.RawArray("B", int(np.prod(image_shape)))
        self._images = np.frombuffer(self._buffer, dtype=np.uint8).reshape(image_shape)
        self._timestamp = ctx.Value("d", 0, lock=False)
        self._prepared_at = ctx.Value("d", 0, lock=False)
        self._lock = ctx.Lock()
        self._request_lock = threading.Lock()
        self._request, self._ready = ctx.Event(), ctx.Event()
        self._stop = ctx.Event()
        self._error = ctx.Array("c", 1024, lock=False)
        self._process = ctx.Process(
            target=_decode_worker,
            args=(
                self._buffer,
                self._timestamp,
                self._prepared_at,
                self._lock,
                self._request,
                self._ready,
                self._stop,
                self._error,
                os.getpid(),
                port,
                payload,
                decoder,
                self._views,
            ),
            name="teleavatar-policy-video",
            daemon=True,
        )

    def start(self):
        self._process.start()

    def stop(self):
        self._stop.set()
        self._request.set()
        if self._process.pid is not None:
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1)

    def stream_ended(self):
        return (
            self._stop.is_set()
            or bool(self._error.value)
            or (self._process.pid is not None and not self._process.is_alive())
        )

    def get_image_timestamps(self):
        timestamp = self._timestamp.value
        return dict.fromkeys(self._views, timestamp) if timestamp else {}

    def get_latest_images_with_timestamps(self):
        # RTC calls this on its inference worker only for a real request.
        # PIL stays in the video process, off the appsink callback.
        with self._request_lock:
            if self.stream_ended():
                raise RuntimeError(self._error.value.decode("utf-8", "replace") or "Policy video process exited")
            if not self._timestamp.value:
                raise RuntimeError("Policy video has no frame")
            self._ready.clear()
            self._request.set()
            if not self._ready.wait(timeout=1):
                self._error.value = b"Policy video preprocessing timed out"
                self._stop.set()
                raise RuntimeError(self._error.value.decode())
            return self._read_prepared_images()

    def _read_prepared_images(self):
        # Bound the wait even if the worker dies while holding its lock.
        if not self._lock.acquire(timeout=0.05):
            raise RuntimeError("Policy video shared-frame lock timed out")
        try:
            if self.stream_ended():
                raise RuntimeError(self._error.value.decode("utf-8", "replace") or "Policy video process exited")
            timestamp = self._prepared_at.value
            if not timestamp:
                raise RuntimeError("Policy video has no frame")
            images = self._images.copy()
        finally:
            self._lock.release()
        return dict(zip(self._views, images, strict=True)), dict.fromkeys(self._views, timestamp)
