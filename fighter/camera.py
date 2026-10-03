"""Background camera + MediaPipe Pose worker.

The game thread never waits on this. The worker publishes an immutable
CameraFrame snapshot after every processed frame, and the game thread just reads
whichever snapshot is newest via `latest()`, so a slow camera or slow pose
model only lowers the tracking rate and never the render rate.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp  # imported up front: importing inside the thread would stall rendering
import numpy as np

from . import config as C


@dataclass(frozen=True)
class CameraFrame:
    frame_id: int
    image: np.ndarray                 # RGB, mirrored, darkened, already screen-sized
    landmarks: Optional[np.ndarray]   # (33, 4): x, y (0..1 of image), z, visibility
    capture_time: float               # time.perf_counter() when the frame was grabbed
    process_ms: float                 # MediaPipe inference time


def _crop_to_aspect(frame: np.ndarray, aspect: float) -> np.ndarray:
    """Center-crop so the camera image matches the window aspect ratio."""
    h, w = frame.shape[:2]
    if w / h > aspect:
        new_w = int(h * aspect)
        x0 = (w - new_w) // 2
        return frame[:, x0:x0 + new_w]
    new_h = int(w / aspect)
    y0 = (h - new_h) // 2
    return frame[y0:y0 + new_h, :]


class CameraThread(threading.Thread):
    def __init__(
        self,
        out_size: Tuple[int, int],
        camera_index: int = C.CAMERA_INDEX,
        model_complexity: int = C.POSE_MODEL_COMPLEXITY,
        brightness: float = C.BACKGROUND_BRIGHTNESS,
    ):
        super().__init__(name="CameraThread", daemon=True)
        self.out_size = out_size
        self.camera_index = camera_index
        self.model_complexity = model_complexity
        self.brightness = brightness

        self._lock = threading.Lock()
        self._latest: Optional[CameraFrame] = None
        self._stop_event = threading.Event()

        self.error: Optional[str] = None
        self.camera_fps = 0.0

    # --- called from the game thread -----------------------------------------
    def latest(self) -> Optional[CameraFrame]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop_event.set()

    # --- worker --------------------------------------------------------------
    def run(self) -> None:
        try:
            self._loop()
        except Exception as exc:  # surfaced on screen by the game thread
            self.error = f"{type(exc).__name__}: {exc}"

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened() and self.camera_index != 0:
            cap.release()
            cap = cv2.VideoCapture(0)  # requested camera missing: use the default one
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.camera_index}. On macOS, grant camera "
                "access to your terminal / VS Code in System Settings > Privacy & Security."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAMERA_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAMERA_H)
        cap.set(cv2.CAP_PROP_FPS, C.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low where supported
        return cap

    def _make_pose(self):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This mediapipe build has no legacy 'solutions' API. "
                "Install a compatible version: pip install mediapipe==0.10.21"
            )
        return mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=C.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=C.POSE_MIN_TRACKING_CONFIDENCE,
        )

    def _loop(self) -> None:
        cap = self._open_capture()
        pose = self._make_pose()
        out_w, out_h = self.out_size
        aspect = out_w / out_h
        pose_size = (C.POSE_INPUT_WIDTH, int(C.POSE_INPUT_WIDTH / aspect))

        frame_id = 0
        failures = 0
        fps_t0, fps_frames = time.perf_counter(), 0
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()  # blocks this thread only
                if not ok or frame is None:
                    failures += 1
                    if failures > 100:
                        raise RuntimeError("Camera stopped delivering frames.")
                    time.sleep(0.01)
                    continue
                failures = 0
                capture_time = time.perf_counter()

                frame = cv2.flip(frame, 1)  # mirror so it behaves like a mirror
                frame = _crop_to_aspect(frame, aspect)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                small = cv2.resize(rgb, pose_size, interpolation=cv2.INTER_AREA)
                small.flags.writeable = False
                t0 = time.perf_counter()
                result = pose.process(small)
                process_ms = (time.perf_counter() - t0) * 1000.0

                landmarks = None
                if result.pose_landmarks:
                    landmarks = np.array(
                        [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks.landmark],
                        dtype=np.float32,
                    )

                # Prepare the background here so the game thread only does a memcpy.
                bg = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                if self.brightness < 1.0:
                    bg = cv2.convertScaleAbs(bg, alpha=self.brightness, beta=0)

                frame_id += 1
                snapshot = CameraFrame(frame_id, bg, landmarks, capture_time, process_ms)
                with self._lock:
                    self._latest = snapshot

                fps_frames += 1
                elapsed = time.perf_counter() - fps_t0
                if elapsed >= 1.0:
                    self.camera_fps = fps_frames / elapsed
                    fps_t0, fps_frames = time.perf_counter(), 0
        finally:
            cap.release()
            pose.close()
