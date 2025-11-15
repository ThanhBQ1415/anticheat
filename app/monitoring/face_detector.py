import logging
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_face(self, frame: np.ndarray) -> Tuple[bool, Optional[dict]]:
        """
        Detect face in frame.
        Returns: (is_face_present, face_data)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            face_data = {
                "x": int(bbox.xmin * w),
                "y": int(bbox.ymin * h),
                "width": int(bbox.width * w),
                "height": int(bbox.height * h),
                "confidence": detection.score[0],
            }
            logger.info(
                "FaceDetector: face detected (x=%s, y=%s, w=%s, h=%s, confidence=%.2f)",
                face_data["x"],
                face_data["y"],
                face_data["width"],
                face_data["height"],
                face_data["confidence"],
            )
            return True, face_data

        logger.debug("FaceDetector: no face detected in current frame")
        return False, None

    def draw_face_box(self, frame: np.ndarray, face_data: dict) -> np.ndarray:
        """Draw bounding box around detected face."""
        cv2.rectangle(
            frame,
            (face_data["x"], face_data["y"]),
            (
                face_data["x"] + face_data["width"],
                face_data["y"] + face_data["height"],
            ),
            (0, 255, 0),
            2,
        )
        return frame

    def release(self):
        """Release resources."""
        self.face_detection.close()
        logger.debug("FaceDetector resources released")

