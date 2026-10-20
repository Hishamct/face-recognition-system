from pathlib import Path
from fastapi.testclient import TestClient

from app.core.config import ROOT_DIR
from app.main import app

client = TestClient(app)

SAMPLE_IMAGE_PATH = ROOT_DIR / "data/processed/aligned_test/Colin_Powell/Colin_Powell_0009.jpg"


def test_list_identities_pagination():
    response = client.get("/api/v1/identities?page=1&page_size=5")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert data["total"] >= 150
    assert data["page"] == 1
    assert data["page_size"] == 5
    assert len(data["items"]) == 5
    first_item = data["items"][0]
    assert "id" in first_item
    assert "name" in first_item
    assert "embedding_count" in first_item
    assert first_item["embedding_count"] >= 1


def test_get_identity_by_id():
    # First get list to find a valid id
    list_res = client.get("/api/v1/identities?page=1&page_size=1")
    assert list_res.status_code == 200
    items = list_res.json()["items"]
    assert len(items) > 0
    valid_id = items[0]["id"]

    response = client.get(f"/api/v1/identities/{valid_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == valid_id
    assert "name" in data
    assert data["embedding_count"] >= 1


def test_get_nonexistent_identity():
    response = client.get("/api/v1/identities/9999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_enroll_and_delete_identity_lifecycle():
    assert SAMPLE_IMAGE_PATH.exists()
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        file_bytes = f.read()

    test_name = "Automated_Test_Person_Phase5"

    # 1. Enroll identity
    enroll_res = client.post(
        "/api/v1/identities",
        data={"name": test_name, "notes": "Temporary test person"},
        files={"file": ("face.jpg", file_bytes, "image/jpeg")},
    )
    assert enroll_res.status_code == 201
    enroll_data = enroll_res.json()
    assert enroll_data["success"] is True
    assert enroll_data["name"] == test_name
    identity_id = enroll_data["identity_id"]
    assert identity_id > 0

    # 2. Verify identity can be fetched
    get_res = client.get(f"/api/v1/identities/{identity_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == test_name

    # 3. Delete identity
    del_res = client.delete(f"/api/v1/identities/{identity_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # 4. Verify identity is deleted
    get_res_after = client.get(f"/api/v1/identities/{identity_id}")
    assert get_res_after.status_code == 404
