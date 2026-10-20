"""
backend/app/vision/video.py
===========================
Video processing engine for frame-by-frame face recognition and tracking.
Supports local video files, streaming RTSP/HTTP endpoints, and webcam feeds.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from app.vision.detector import FaceDetection, YuNetDetector
    from app.vision.aligner import FaceAligner
    from app.vision.embedder import FaceEmbedder
    from app.vision.classifier import IdentityClassifier
    from app.schemas.recognition import FaceRecognitionItem, CandidateItem
    from app.schemas.common import BoundingBox, Landmarks5, LandmarkPoint
except ImportError:
    from backend.app.vision.detector import FaceDetection, YuNetDetector
    from backend.app.vision.aligner import FaceAligner
    from backend.app.vision.embedder import FaceEmbedder
    from backend.app.vision.classifier import IdentityClassifier
    from backend.app.schemas.recognition import FaceRecognitionItem, CandidateItem
    from backend.app.schemas.common import BoundingBox, Landmarks5, LandmarkPoint


@dataclass
class VideoFrameResult:
    """Telemetry and recognition results for a single video frame."""
    frame_index: int
    timestamp_ms: float
    face_count: int
    faces: List[FaceRecognitionItem]
    inference_time_ms: float
    annotated_frame: Optional[np.ndarray] = None


class VideoProcessor:
    """
    High-performance video stream and file processor.
    Extracts frames, performs detection + alignment + recognition, and calculates throughput (FPS).
    """

    def __init__(
        self,
        detector: Optional[YuNetDetector] = None,
        aligner: Optional[FaceAligner] = None,
        embedder: Optional[FaceEmbedder] = None,
        classifier: Optional[IdentityClassifier] = None,
        threshold: float = 0.5697,
    ):
        self.detector = detector or YuNetDetector()
        self.aligner = aligner or FaceAligner()
        self.embedder = embedder or FaceEmbedder()
        self.classifier = classifier
        self.threshold = threshold

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        annotate: bool = False,
    ) -> VideoFrameResult:
        """
        Runs the full recognition pipeline on a single BGR video frame.
        """
        t0 = time.perf_counter()
        if frame is None or frame.size == 0:
            return VideoFrameResult(
                frame_index=frame_index,
                timestamp_ms=0.0,
                face_count=0,
                faces=[],
                inference_time_ms=0.0,
            )

        detections = self.detector.detect(frame)
        face_items: List[FaceRecognitionItem] = []

        for det in detections:
            # 1. Canonical 112x112 Alignment
            aligned_face = self.aligner.align(frame, det.landmarks)

            # 2. 128-D Embedding
            embedding = self.embedder.extract(aligned_face)

            # 3. Predict Identity
            top_identity = "Unknown"
            confidence = 0.0
            top_candidates: List[CandidateItem] = []

            if self.classifier is not None:
                name, prob, cands = self.classifier.predict_embedding(embedding, top_k=3)
                top_identity = name
                confidence = prob
                for c in cands:
                    top_candidates.append(
                        CandidateItem(
                            identity=str(c["identity"]),
                            confidence=round(float(c["probability"]), 4),
                            rank=int(c["rank"]),
                        )
                    )
            else:
                top_identity = "Unknown"
                confidence = float(det.score)

            # 4. Decision Rule
            is_recognized = (
                confidence >= self.threshold and top_identity.lower() != "unknown"
            )
            final_name = top_identity if is_recognized else "Unknown"

            landmarks_dict = {
                "left_eye": LandmarkPoint(x=float(det.landmarks[0][0]), y=float(det.landmarks[0][1])),
                "right_eye": LandmarkPoint(x=float(det.landmarks[1][0]), y=float(det.landmarks[1][1])),
                "nose_tip": LandmarkPoint(x=float(det.landmarks[2][0]), y=float(det.landmarks[2][1])),
                "mouth_left": LandmarkPoint(x=float(det.landmarks[3][0]), y=float(det.landmarks[3][1])),
                "mouth_right": LandmarkPoint(x=float(det.landmarks[4][0]), y=float(det.landmarks[4][1])),
            }

            bbox_obj = BoundingBox(
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
                width=det.bbox[2],
                height=det.bbox[3],
                confidence=round(float(det.score), 4),
            )

            face_items.append(
                FaceRecognitionItem(
                    bbox=bbox_obj,
                    landmarks=Landmarks5(**landmarks_dict),
                    identity=final_name,
                    confidence=round(float(confidence), 4),
                    status="RECOGNIZED" if is_recognized else "UNKNOWN",
                    is_known=is_recognized,
                    threshold_used=self.threshold,
                    top_candidates=top_candidates,
                )
            )

        inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        annotated = None
        if annotate:
            annotated = self.annotate_frame(frame, face_items)

        return VideoFrameResult(
            frame_index=frame_index,
            timestamp_ms=round(frame_index * (1000.0 / 30.0), 2),
            face_count=len(face_items),
            faces=face_items,
            inference_time_ms=inference_time_ms,
            annotated_frame=annotated,
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        faces: List[FaceRecognitionItem],
    ) -> np.ndarray:
        """
        Draws technical bounding boxes, corner brackets, and labels on a frame copy.
        """
        out = frame.copy()
        for f in faces:
            x1, y1 = f.bbox.x1, f.bbox.y1
            x2, y2 = f.bbox.x2, f.bbox.y2
            color = (79, 70, 229) if f.status == "RECOGNIZED" else (11, 158, 245) # BGR: Indigo / Amber

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label text
            label = f"{f.identity} ({int(f.confidence * 100)}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (15, 23, 42), -1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, 1)
            cv2.putText(
                out,
                label,
                (x1 + 4, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # 5 landmarks
            for pt in [
                f.landmarks.left_eye,
                f.landmarks.right_eye,
                f.landmarks.nose_tip,
                f.landmarks.mouth_left,
                f.landmarks.mouth_right,
            ]:
                cv2.circle(out, (int(pt.x), int(pt.y)), 3, color, -1)
                cv2.circle(out, (int(pt.x), int(pt.y)), 1, (255, 255, 255), -1)

        return out

    def process_video_file(
        self,
        video_path: Union[str, Path],
        frame_stride: int = 2,
        max_frames: Optional[int] = 300,
    ) -> Generator[VideoFrameResult, None, None]:
        """
        Processes a video file frame-by-frame and yields telemetry results.
        
        Args:
            video_path: Path to .mp4, .avi, or .mov file
            frame_stride: Process every Nth frame (e.g. 2 = 15 FPS on 30 FPS video)
            max_frames: Max frames to process before stopping
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {video_path}")

        frame_idx = 0
        processed_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_stride == 0:
                    result = self.process_frame(frame, frame_index=frame_idx)
                    yield result
                    processed_count += 1
                    if max_frames and processed_count >= max_frames:
                        break

                frame_idx += 1
        finally:
            cap.release()
