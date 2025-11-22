import logging
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

logger = logging.getLogger(__name__)


class EyeTracker:
    LEFT_EYE_CORNER_INDICES = (33, 133)         #góc mắt trái
    LEFT_IRIS_INDICES = (468, 469, 470, 471)    #điểm landmark con ngươi

    RIGHT_EYE_CORNER_INDICES = (362, 263)        #góc mắt phải
    RIGHT_IRIS_INDICES = (473, 474, 475, 476)    #điểm landmark con ngươi

    #khởi tạo và thiết lập tham số
    def __init__(self, predictor_path: Optional[str] = None):
        if predictor_path:
            logger.warning(
                "EyeTracker: predictor_path argument is ignored in MediaPipe pipeline"
            )

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,  
            max_num_faces=1,        
            refine_landmarks=True,   
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5, 
        )

        self.CONSECUTIVE_LOOK_AWAY_FRAMES = 5
        self.GAZE_MIN_THRESHOLD = 0.4 
        self.GAZE_MAX_THRESHOLD = 0.6 

        self.consecutive_look_away_count = 0 

        logger.info("EyeTracker: using MediaPipe FaceMesh with iris landmarks") 

    def _compute_eye_ratio(
        self,
        face_landmarks: landmark_pb2.NormalizedLandmarkList,
        corner_indices: Tuple[int, int],
        iris_indices: Tuple[int, int, int, int],
        flip_horizontal: bool,
    ) -> Optional[float]:
        landmarks = face_landmarks.landmark

        try:
            corners_x = [landmarks[idx].x for idx in corner_indices]
            iris_points_x = [landmarks[idx].x for idx in iris_indices]
        except IndexError:
            return None

        if not iris_points_x:
            return None

        iris_center_x = float(np.mean(iris_points_x))
        min_corner = float(np.min(corners_x))
        max_corner = float(np.max(corners_x))
        denominator = max_corner - min_corner
        if denominator <= 1e-6:
            return None

        ratio = (iris_center_x - min_corner) / denominator
        if flip_horizontal:
            ratio = 1.0 - ratio
        return float(np.clip(ratio, 0.0, 1.0))




    #tính toán tỉ lệ iris trung bình 2 mắt
    def _get_iris_ratio(
        self, face_landmarks: landmark_pb2.NormalizedLandmarkList
    ) -> Optional[Tuple[float, float]]:
        left_ratio = self._compute_eye_ratio(
            face_landmarks,
            self.LEFT_EYE_CORNER_INDICES,
            self.LEFT_IRIS_INDICES,
            flip_horizontal=False,
        )
        right_ratio = self._compute_eye_ratio(
            face_landmarks,
            self.RIGHT_EYE_CORNER_INDICES,
            self.RIGHT_IRIS_INDICES,
            flip_horizontal=True,
        )

        if left_ratio is None or right_ratio is None:
            return None
        return (left_ratio, right_ratio)

    def is_looking_away(self, frame: np.ndarray) -> Tuple[bool, Optional[str]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            self._reset_look_away_state()
            return False, None

        face_landmarks = results.multi_face_landmarks[0]
        ratios = self._get_iris_ratio(face_landmarks)
        
        if ratios is None:
            self.consecutive_look_away_count = 0
            return False, None

        left_ratio, right_ratio = ratios
        logger.info(f"left_ratio: {left_ratio}, right_ratio: {right_ratio}")
        
        left_in_range = self.GAZE_MIN_THRESHOLD < left_ratio < self.GAZE_MAX_THRESHOLD
        right_in_range = self.GAZE_MIN_THRESHOLD < right_ratio < self.GAZE_MAX_THRESHOLD
    
        if left_in_range or right_in_range:
            is_looking_away = False
            logger.debug(
                "EyeTracker: at least one eye in range (left=%.3f, right=%.3f)",
                left_ratio,
                right_ratio,
            )
        else:
            is_looking_away = True
            logger.info(
                "EyeTracker: both eyes out of range (left=%.3f, right=%.3f)",
                left_ratio,
                right_ratio,
            )
        if is_looking_away:
            self.consecutive_look_away_count += 1
            logger.info(
                "EyeTracker: look-away detected (count=%d/%d)",
                self.consecutive_look_away_count,
                self.CONSECUTIVE_LOOK_AWAY_FRAMES,
            )
            if self.consecutive_look_away_count >= self.CONSECUTIVE_LOOK_AWAY_FRAMES:
                message = f"Looking away detected ({self.consecutive_look_away_count} consecutive frames)"
                logger.info("EyeTracker: confirmed look-away violation")
                self.consecutive_look_away_count = 0 
                return True, message
        else:
            if self.consecutive_look_away_count > 0:
                logger.debug("EyeTracker: gaze back on screen. Resetting counter.")
            self.consecutive_look_away_count = 0

        return False, None

    def _reset_look_away_state(self) -> None:
        self.consecutive_look_away_count = 0

    def release(self) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None
        logger.debug("EyeTracker: release called, MediaPipe resources released")

