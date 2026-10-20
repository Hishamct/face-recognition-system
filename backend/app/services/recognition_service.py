import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from sqlalchemy.orm import Session

try:
    from app.core.config import ROOT_DIR, settings
    from app.db.models import RecognitionAuditLog
    from app.schemas.common import BoundingBox, LandmarkPoint, Landmarks5
    from app.schemas.detection import DetectionResponse, FaceDetectionItem
    from app.schemas.recognition import CandidateItem, FaceRecognitionItem, RecognitionAuditItem, RecognitionResponse
    from app.services.vector_store import PostgresVectorStore
    from app.vision.aligner import FaceAligner
    from app.vision.classifier import IdentityClassifier
    from app.vision.detector import YuNetDetector
    from app.vision.embedder import FaceEmbedder
except ImportError:
    from backend.app.core.config import ROOT_DIR, settings
    from backend.app.db.models import RecognitionAuditLog
    from backend.app.schemas.common import BoundingBox, LandmarkPoint, Landmarks5
    from backend.app.schemas.detection import DetectionResponse, FaceDetectionItem
    from backend.app.schemas.recognition import CandidateItem, FaceRecognitionItem, RecognitionAuditItem, RecognitionResponse
    from backend.app.services.vector_store import PostgresVectorStore
    from backend.app.vision.aligner import FaceAligner
    from backend.app.vision.classifier import IdentityClassifier
    from backend.app.vision.detector import YuNetDetector
    from backend.app.vision.embedder import FaceEmbedder


def _resolve_model_path(relative_path: str) -> Path:
    """Finds model path either from CWD or from project ROOT_DIR."""
    p = Path(relative_path)
    if p.exists():
        return p
    p_root = ROOT_DIR / relative_path
    if p_root.exists():
        return p_root
    return p


class RecognitionService:
    """
    Central vision orchestrator for Face Detection, Alignment,
    Embedding extraction, Classification, and Relational Audit Logging.
    """

    # Calibrated operating thresholds from Phase 3 (cosine) and Phase 4 (classifier)
    DEFAULT_CLASSIFIER_THRESHOLD: float = 0.5697
    DEFAULT_COSINE_THRESHOLD: float = 0.445

    def __init__(self):
        self._detector: Optional[YuNetDetector] = None
        self._aligner: Optional[FaceAligner] = None
        self._embedder: Optional[FaceEmbedder] = None
        self._classifier: Optional[IdentityClassifier] = None
        self._vector_store: Optional[PostgresVectorStore] = None

    @property
    def detector(self) -> YuNetDetector:
        if self._detector is None:
            yunet_path = _resolve_model_path("models/face_detection_yunet_2023mar.onnx")
            self._detector = YuNetDetector(model_path=str(yunet_path), score_threshold=0.6)
        return self._detector

    @property
    def aligner(self) -> FaceAligner:
        if self._aligner is None:
            self._aligner = FaceAligner()
        return self._aligner

    @property
    def embedder(self) -> FaceEmbedder:
        if self._embedder is None:
            sface_path = _resolve_model_path("models/face_recognition_sface_2021dec.onnx")
            self._embedder = FaceEmbedder(model_path=str(sface_path))
        return self._embedder

    @property
    def classifier(self) -> Optional[IdentityClassifier]:
        if self._classifier is None:
            weights_path = _resolve_model_path("models/classifier_head.pt")
            if weights_path.exists():
                try:
                    self._classifier = IdentityClassifier.load(
                        filepath=str(weights_path),
                        device="cpu"
                    )
                except Exception as exc:
                    print(f"Warning: could not load classifier head: {exc}")
                    self._classifier = None
        return self._classifier

    @property
    def vector_store(self) -> PostgresVectorStore:
        if self._vector_store is None:
            self._vector_store = PostgresVectorStore()
        return self._vector_store

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decodes raw uploaded image bytes into a BGR OpenCV image."""
        if not image_bytes:
            raise ValueError("Uploaded image file is empty.")
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("Could not decode image. Supported formats: JPEG, PNG, WEBP, BMP.")
        return image

    def detect_faces(self, image_bytes: bytes) -> DetectionResponse:
        """Runs face detection and returns bounding boxes and 5-point facial landmarks."""
        t0 = time.perf_counter()
        image = self.decode_image(image_bytes)
        h, w = image.shape[:2]

        raw_detections = self.detector.detect(image)
        faces: List[FaceDetectionItem] = []

        for d in raw_detections:
            landmarks_dict = {
                "left_eye": LandmarkPoint(x=float(d.landmarks[0][0]), y=float(d.landmarks[0][1])),
                "right_eye": LandmarkPoint(x=float(d.landmarks[1][0]), y=float(d.landmarks[1][1])),
                "nose_tip": LandmarkPoint(x=float(d.landmarks[2][0]), y=float(d.landmarks[2][1])),
                "mouth_left": LandmarkPoint(x=float(d.landmarks[3][0]), y=float(d.landmarks[3][1])),
                "mouth_right": LandmarkPoint(x=float(d.landmarks[4][0]), y=float(d.landmarks[4][1])),
            }
            faces.append(
                FaceDetectionItem(
                    bbox=BoundingBox(
                        x1=d.x1,
                        y1=d.y1,
                        x2=d.x2,
                        y2=d.y2,
                        width=d.bbox[2],
                        height=d.bbox[3],
                        confidence=round(float(d.score), 4),
                    ),
                    landmarks=Landmarks5(**landmarks_dict),
                    confidence=round(float(d.score), 4),
                )
            )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return DetectionResponse(
            success=True,
            face_count=len(faces),
            faces=faces,
            image_width=w,
            image_height=h,
            latency_ms=latency_ms,
        )

    def recognize_faces(
        self,
        image_bytes: bytes,
        threshold_override: Optional[float] = None,
        use_classifier: bool = True,
        db: Optional[Session] = None,
    ) -> RecognitionResponse:
        """
        Runs the full biometric pipeline:
        Detection -> 5-Point Affine Alignment -> SFace 128-D Embedding -> Classification/Matching.
        """
        t0 = time.perf_counter()
        image = self.decode_image(image_bytes)
        h, w = image.shape[:2]

        raw_detections = self.detector.detect(image)
        face_items: List[FaceRecognitionItem] = []
        recognized_count = 0
        unknown_count = 0

        # Choose threshold and method
        classifier_model = self.classifier if use_classifier else None
        if classifier_model is not None:
            threshold = (
                float(threshold_override)
                if threshold_override is not None
                else self.DEFAULT_CLASSIFIER_THRESHOLD
            )
        else:
            threshold = (
                float(threshold_override)
                if threshold_override is not None
                else self.DEFAULT_COSINE_THRESHOLD
            )

        for d in raw_detections:
            # 1. Canonical 112x112 Alignment
            aligned_face = self.aligner.align(image, d.landmarks)

            # 2. Extract 128-D L2-normalized SFace embedding
            embedding = self.embedder.extract(aligned_face)

            # 3. Predict Identity
            top_candidates: List[CandidateItem] = []
            if classifier_model is not None:
                # Use supervised PyTorch classifier head
                top1_name, top1_prob, candidate_dicts = classifier_model.predict_embedding(embedding, top_k=5)
                top_identity = top1_name
                top_conf = top1_prob
                for c in candidate_dicts:
                    top_candidates.append(
                        CandidateItem(
                            identity=str(c["identity"]),
                            confidence=round(float(c["probability"]), 4),
                            rank=int(c["rank"]),
                        )
                    )
            else:
                # Fallback to vector search
                matches = self.vector_store.search(embedding, k=5)
                if matches:
                    top_identity = matches[0]["identity_name"]
                    top_conf = matches[0]["similarity"]
                    for rank_idx, m in enumerate(matches, start=1):
                        top_candidates.append(
                            CandidateItem(
                                identity=m["identity_name"],
                                confidence=round(float(m["similarity"]), 4),
                                rank=rank_idx,
                            )
                        )
                else:
                    top_identity = "Unknown"
                    top_conf = 0.0

            # 4. Decision Rule
            if top_conf >= threshold:
                status = "RECOGNIZED"
                identity_label = top_identity
                recognized_count += 1
            else:
                status = "UNKNOWN"
                identity_label = "Unknown"
                unknown_count += 1

            landmarks_dict = {
                "left_eye": LandmarkPoint(x=float(d.landmarks[0][0]), y=float(d.landmarks[0][1])),
                "right_eye": LandmarkPoint(x=float(d.landmarks[1][0]), y=float(d.landmarks[1][1])),
                "nose_tip": LandmarkPoint(x=float(d.landmarks[2][0]), y=float(d.landmarks[2][1])),
                "mouth_left": LandmarkPoint(x=float(d.landmarks[3][0]), y=float(d.landmarks[3][1])),
                "mouth_right": LandmarkPoint(x=float(d.landmarks[4][0]), y=float(d.landmarks[4][1])),
            }

            face_items.append(
                FaceRecognitionItem(
                    bbox=BoundingBox(
                        x1=d.x1,
                        y1=d.y1,
                        x2=d.x2,
                        y2=d.y2,
                        width=d.bbox[2],
                        height=d.bbox[3],
                        confidence=round(float(d.score), 4),
                    ),
                    landmarks=Landmarks5(**landmarks_dict),
                    identity=identity_label,
                    confidence=round(float(top_conf), 4),
                    status=status,
                    threshold_used=round(threshold, 4),
                    top_candidates=top_candidates,
                )
            )

            # 5. Audit Logging to PostgreSQL
            if db is not None:
                try:
                    audit = RecognitionAuditLog(
                        matched_identity=top_identity if status == "RECOGNIZED" else "Unknown",
                        similarity_score=float(top_conf),
                        threshold_used=float(threshold),
                        decision=status,
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                    )
                    db.add(audit)
                    db.commit()
                except Exception as log_exc:
                    db.rollback()
                    print(f"Warning: failed to commit audit log: {log_exc}")

        total_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return RecognitionResponse(
            success=True,
            face_count=len(face_items),
            recognized_count=recognized_count,
            unknown_count=unknown_count,
            faces=face_items,
            image_width=w,
            image_height=h,
            latency_ms=total_latency_ms,
        )

    def get_audit_logs(self, db: Session, limit: int = 50) -> List[RecognitionAuditItem]:
        """Retrieves recent biometric recognition audit log entries."""
        logs = (
            db.query(RecognitionAuditLog)
            .order_by(RecognitionAuditLog.query_timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            RecognitionAuditItem(
                id=log.id,
                matched_identity=log.matched_identity,
                similarity_score=round(float(log.similarity_score), 4),
                threshold_used=round(float(log.threshold_used), 4),
                decision=log.decision,
                query_latency_ms=round(float(log.latency_ms), 2),
                created_at=log.query_timestamp,
            )
            for log in logs
        ]


# Singleton instance
recognition_service = RecognitionService()
