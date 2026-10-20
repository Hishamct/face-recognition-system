from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.core.config import ROOT_DIR
from app.main import app

client = TestClient(app)

SAMPLE_IMAGE_PATH = ROOT_DIR / "data/processed/aligned_test/Colin_Powell/Colin_Powell_0009.jpg"


def test_detect_faces_real_image():
    assert SAMPLE_IMAGE_PATH.exists(), f"Sample image not found at {SAMPLE_IMAGE_PATH}"
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/detection/image",
        files={"file": ("colin.jpg", file_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["face_count"] >= 1
    assert "latency_ms" in data
    assert len(data["faces"]) >= 1

    first_face = data["faces"][0]
    bbox = first_face["bbox"]
    assert "x1" in bbox and "y1" in bbox and "width" in bbox and "height" in bbox
    assert bbox["confidence"] >= 0.6
    landmarks = first_face["landmarks"]
    assert "left_eye" in landmarks and "right_eye" in landmarks and "nose_tip" in landmarks


def test_detect_faces_blank_image():
    # 200x200 black image
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", blank)

    response = client.post(
        "/api/v1/detection/image",
        files={"file": ("blank.jpg", encoded.tobytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["face_count"] == 0
    assert len(data["faces"]) == 0


def test_detect_faces_empty_file():
    response = client.post(
        "/api/v1/detection/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_detect_faces_invalid_content_type():
    response = client.post(
        "/api/v1/detection/image",
        files={"file": ("test.txt", b"plain text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "invalid file type" in response.json()["detail"].lower()
