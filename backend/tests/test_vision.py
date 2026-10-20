"""
backend/tests/test_vision.py
============================
Automated unit tests for Phase 2 computer vision modules:
YuNet face detection, 5-point landmark extraction, and affine alignment.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from backend.app.vision.detector import YuNetDetector, FaceDetection
from backend.app.vision.aligner import FaceAligner, ARCFACE_REFERENCE_112


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def detector(repo_root: Path) -> YuNetDetector:
    model_path = repo_root / "models" / "face_detection_yunet_2023mar.onnx"
    assert model_path.exists(), "YuNet model must exist in models/"
    return YuNetDetector(model_path=model_path, score_threshold=0.7)


@pytest.fixture
def sample_face_image(repo_root: Path) -> np.ndarray:
    sample_files = list((repo_root / "data" / "processed" / "train").rglob("*.jpg"))
    if not sample_files:
        pytest.skip("Dataset not present for sample image fixture")
    img = cv2.imread(str(sample_files[0]))
    assert img is not None
    return img


def test_detector_initialization(detector: YuNetDetector):
    assert detector is not None
    assert detector.score_threshold == 0.7
    assert detector.nms_threshold == 0.3


def test_detector_on_real_face(detector: YuNetDetector, sample_face_image: np.ndarray):
    detections = detector.detect(sample_face_image)
    assert len(detections) >= 1, "Must detect at least 1 face in valid LFW photo"

    det = detections[0]
    assert isinstance(det, FaceDetection)
    assert det.score >= 0.7
    assert len(det.bbox) == 4
    assert det.bbox[2] > 20 and det.bbox[3] > 20, "Bounding box must have reasonable size"
    assert det.landmarks.shape == (5, 2), "Landmarks must be 5 (x, y) coordinates"


def test_detector_on_blank_image(detector: YuNetDetector):
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = detector.detect(blank)
    assert detections == [], "Blank black image should contain 0 detections"


def test_detector_draw_detections(detector: YuNetDetector, sample_face_image: np.ndarray):
    detections = detector.detect(sample_face_image)
    annotated = detector.draw_detections(sample_face_image, detections)
    assert annotated.shape == sample_face_image.shape
    assert annotated.dtype == np.uint8


def test_aligner_canonical_output():
    aligner = FaceAligner(target_size=(112, 112))
    assert aligner.target_size == (112, 112)
    assert aligner.reference_landmarks.shape == (5, 2)

    # Synthetic image and landmarks
    img = np.full((250, 250, 3), 100, dtype=np.uint8)
    # Landmark coordinates roughly matching reference scaled to 250
    landmarks = ARCFACE_REFERENCE_112 * (250.0 / 112.0)

    aligned = aligner.align(img, landmarks)
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8


def test_aligner_on_detected_face(detector: YuNetDetector, sample_face_image: np.ndarray):
    detections = detector.detect(sample_face_image)
    assert len(detections) >= 1

    aligner = FaceAligner(target_size=(112, 112))
    aligned = aligner.align(sample_face_image, detections[0].landmarks, fallback_bbox=detections[0].bbox)

    assert aligned.shape == (112, 112, 3), "Aligned output must be 112x112x3"
    assert aligned.dtype == np.uint8
    assert aligned.std() > 5.0, "Aligned face should contain valid pixel texture, not blank color"
