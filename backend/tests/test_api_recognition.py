from pathlib import Path
from fastapi.testclient import TestClient

from app.core.config import ROOT_DIR
from app.main import app

client = TestClient(app)

SAMPLE_IMAGE_PATH = ROOT_DIR / "data/processed/aligned_test/Colin_Powell/Colin_Powell_0009.jpg"


def test_recognize_faces_real_image():
    assert SAMPLE_IMAGE_PATH.exists(), f"Sample image not found at {SAMPLE_IMAGE_PATH}"
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/recognition/image",
        files={"file": ("colin.jpg", file_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["face_count"] >= 1
    assert data["recognized_count"] + data["unknown_count"] == data["face_count"]

    first_face = data["faces"][0]
    assert "identity" in first_face
    assert first_face["identity"] == "Colin_Powell"
    assert first_face["status"] == "RECOGNIZED"
    assert first_face["confidence"] >= 0.5
    assert len(first_face["top_candidates"]) >= 1
    assert first_face["top_candidates"][0]["identity"] == "Colin_Powell"


def test_recognize_faces_with_strict_threshold_override():
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        file_bytes = f.read()

    # Pass threshold 1.0 which exceeds the model's confidence
    response = client.post(
        "/api/v1/recognition/image?threshold=1.0",
        files={"file": ("colin.jpg", file_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["faces"][0]["status"] == "UNKNOWN"
    assert data["faces"][0]["identity"] == "Unknown"
    assert data["unknown_count"] >= 1


def test_get_recognition_audit_logs():
    response = client.get("/api/v1/recognition/audit-logs?limit=10")
    assert response.status_code == 200
    logs = response.json()
    assert isinstance(logs, list)
    if logs:
        first_log = logs[0]
        assert "decision" in first_log
        assert "similarity_score" in first_log
        assert "query_latency_ms" in first_log
