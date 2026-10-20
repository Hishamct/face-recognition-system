import base64
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import ROOT_DIR
from app.main import app
from app.vision.video import VideoProcessor
from app.services.recognition_service import recognition_service

client = TestClient(app)

SAMPLE_IMAGE_PATH = ROOT_DIR / "data/processed/aligned_test/Colin_Powell/Colin_Powell_0009.jpg"


@pytest.fixture
def sample_face_bytes():
    assert SAMPLE_IMAGE_PATH.exists(), f"Sample image missing at {SAMPLE_IMAGE_PATH}"
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        return f.read()


@pytest.fixture
def sample_face_bgr():
    img = cv2.imread(str(SAMPLE_IMAGE_PATH))
    assert img is not None
    return img


def test_video_processor_blank_frame():
    processor = VideoProcessor(
        detector=recognition_service.detector,
        aligner=recognition_service.aligner,
        embedder=recognition_service.embedder,
        classifier=recognition_service.classifier,
    )
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    result = processor.process_frame(blank, frame_index=0)
    assert result.frame_index == 0
    assert result.face_count == 0
    assert len(result.faces) == 0
    assert result.inference_time_ms >= 0


def test_video_processor_real_face_frame(sample_face_bgr):
    processor = VideoProcessor(
        detector=recognition_service.detector,
        aligner=recognition_service.aligner,
        embedder=recognition_service.embedder,
        classifier=recognition_service.classifier,
    )
    result = processor.process_frame(sample_face_bgr, frame_index=1, annotate=True)
    assert result.face_count >= 1
    assert len(result.faces) == result.face_count
    assert result.annotated_frame is not None
    assert result.annotated_frame.shape == sample_face_bgr.shape

    face = result.faces[0]
    assert face.identity == "Colin_Powell"
    assert face.status == "RECOGNIZED"
    assert face.bbox is not None


def test_websocket_video_stream_binary(sample_face_bytes):
    with client.websocket_connect("/api/v1/video/stream") as websocket:
        # Send raw JPEG bytes
        websocket.send_bytes(sample_face_bytes)
        data = websocket.receive_json()

        assert data["success"] is True
        assert data["face_count"] >= 1
        assert "inference_time_ms" in data
        assert "fps" in data
        assert "image_width" in data
        assert "image_height" in data
        assert len(data["faces"]) >= 1

        first_face = data["faces"][0]
        assert first_face["identity"] == "Colin_Powell"
        assert first_face["status"] == "RECOGNIZED"


def test_websocket_video_stream_json_base64_with_threshold_override(sample_face_bytes):
    import json
    b64_str = base64.b64encode(sample_face_bytes).decode("utf-8")
    with client.websocket_connect("/api/v1/video/stream") as websocket:
        # Send JSON with base64 and strict threshold
        payload = {
            "frame": f"data:image/jpeg;base64,{b64_str}",
            "threshold": 1.05,  # Unattainable threshold to force UNKNOWN
        }
        websocket.send_text(json.dumps(payload))
        data = websocket.receive_json()

        assert data["success"] is True
        assert data["face_count"] >= 1
        first_face = data["faces"][0]
        assert first_face["status"] == "UNKNOWN"
        assert first_face["is_known"] is False


def test_video_upload_endpoint(sample_face_bgr):
    # Create a small 5-frame synthetic MP4 video in a temporary file
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, "test_synthetic_clip.mp4")

    h, w = sample_face_bgr.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (w, h))

    try:
        for _ in range(5):
            out.write(sample_face_bgr)
        out.release()

        with open(video_path, "rb") as f:
            file_bytes = f.read()

        response = client.post(
            "/api/v1/video/upload?stride=1&confidence_threshold=0.5",
            files={"file": ("test_clip.mp4", file_bytes, "video/mp4")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "test_clip.mp4"
        assert data["total_frames_processed"] == 5
        assert data["duration_seconds"] > 0
        assert data["average_fps"] > 0
        assert "Colin_Powell" in data["unique_identities_detected"]
        assert len(data["timeline_sample"]) >= 1

    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass
