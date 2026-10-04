"""
VisionPlate Webcam Worker Module
Handles live camera stream processing in a dedicated background QThread
to ensure zero GUI freezing or blocking.
"""
import time
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Slot
from app.detector import PlateDetector


class CameraWorker(QThread):
    """
    Worker QThread for reading live OpenCV webcam frames and running YOLO detection.
    """
    frame_processed = Signal(object, list)  # (annotated_bgr_frame, detections_list)
    error_occurred = Signal(str)            # Error message
    camera_stopped = Signal()              # Camera loop exited

    def __init__(self, detector: PlateDetector, camera_index: int = 0, conf_threshold: float = 0.25):
        super().__init__()
        self.detector = detector
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self._is_running = False
        self.cap = None

    def set_confidence(self, conf: float):
        """Update confidence threshold dynamically while camera is running."""
        self.conf_threshold = max(0.01, min(conf, 1.0))

    def stop(self):
        """Request the camera loop to stop gracefully."""
        self._is_running = False

    def run(self):
        """Thread execution entry point."""
        self._is_running = True
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.error_occurred.emit(f"Webcam unavailable (Device index {self.camera_index}). Please check your camera.")
                self._is_running = False
                self.camera_stopped.emit()
                return

            # Main capture loop
            consecutive_failures = 0
            while self._is_running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        self.error_occurred.emit("Webcam frame capture failed (Device disconnected or unreadable).")
                        break
                    self.msleep(50)
                    continue

                consecutive_failures = 0

                # Run detection on current frame
                result = self.detector.detect(frame, conf_threshold=self.conf_threshold)

                if result["success"]:
                    self.frame_processed.emit(result["annotated_image"], result["detections"])
                else:
                    # Fall back to raw frame if detection failed with soft error
                    self.frame_processed.emit(frame, [])

                # Sleep to maintain smooth ~30 FPS without overloading CPU
                self.msleep(30)

        except Exception as e:
            self.error_occurred.emit(f"Webcam thread exception: {str(e)}")
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self._is_running = False
            self.camera_stopped.emit()
