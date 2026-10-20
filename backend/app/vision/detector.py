"""
backend/app/vision/detector.py
==============================
YuNet Deep Face Detector wrapper utilizing OpenCV's native FaceDetectorYN API.
Outputs high-precision bounding boxes, confidence scores, and 5 facial landmarks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


@dataclass
class FaceDetection:
    """Represents a single detected face."""
    bbox: list[int]          # [x, y, width, height]
    score: float             # Detection confidence score (0.0 to 1.0)
    landmarks: np.ndarray    # (5, 2) array: [right_eye, left_eye, nose_tip, right_mouth, left_mouth]

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[0] + self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[1] + self.bbox[3]


class YuNetDetector:
    """
    Wrapper for OpenCV's native YuNet face detector model.
    Runs fast on CPU (< 30ms on 640x480) and detects 5 facial landmarks.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.75,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YuNet model file not found at: {self.model_path}\n"
                "Please ensure models/face_detection_yunet_2023mar.onnx is downloaded."
            )

        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)

        # Initialize detector with default input size; will be updated dynamically per frame
        self._detector = cv2.FaceDetectorYN.create(
            model=str(self.model_path),
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """
        Detect faces in a BGR or RGB image.
        
        Args:
            image: np.ndarray (H, W, 3) in BGR or RGB format
            
        Returns:
            List of FaceDetection objects, sorted by confidence descending.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        max_dim = 1280
        scale = 1.0
        detect_img = image

        # For high-resolution images (> 1280px), downsample for real-time inference speed
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            detect_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        self._detector.setInputSize((w, h))

        ret, raw_detections = self._detector.detect(detect_img)
        if raw_detections is None or len(raw_detections) == 0:
            return []

        results = []
        for det in raw_detections:
            if scale != 1.0:
                bbox = [
                    int(round(det[0] / scale)),
                    int(round(det[1] / scale)),
                    int(round(det[2] / scale)),
                    int(round(det[3] / scale)),
                ]
                landmarks = (det[4:14].reshape((5, 2)) / scale).astype(np.float32)
            else:
                bbox = [int(det[0]), int(det[1]), int(det[2]), int(det[3])]
                landmarks = det[4:14].reshape((5, 2)).astype(np.float32)

            score = float(det[14])
            results.append(FaceDetection(bbox=bbox, score=score, landmarks=landmarks))

        # Sort by confidence descending
        results.sort(key=lambda d: d.score, reverse=True)
        return results

    def draw_detections(
        self,
        image: np.ndarray,
        detections: list[FaceDetection],
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw bounding boxes, scores, and 5 landmarks on an image copy for debugging.
        """
        annotated = image.copy()
        for det in detections:
            # Draw bounding box
            cv2.rectangle(
                annotated,
                (det.x1, det.y1),
                (det.x2, det.y2),
                color,
                2,
            )

            # Draw confidence label
            label = f"{det.score:.2f}"
            cv2.putText(
                annotated,
                label,
                (det.x1, max(15, det.y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

            # Draw 5 landmark points (colors: eyes=cyan, nose=yellow, mouth=magenta)
            landmark_colors = [
                (255, 255, 0),  # Right eye
                (255, 255, 0),  # Left eye
                (0, 255, 255),  # Nose tip
                (255, 0, 255),  # Right mouth corner
                (255, 0, 255),  # Left mouth corner
            ]
            for (lx, ly), pt_color in zip(det.landmarks, landmark_colors):
                cv2.circle(annotated, (int(lx), int(ly)), 3, pt_color, -1)

        return annotated
