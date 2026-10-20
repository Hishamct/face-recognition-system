from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_get_latest_evaluation_metrics():
    response = client.get("/api/v1/evaluation/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "rank1_acc" in data
    assert data["rank1_acc"] >= 0.95
    assert "top5_acc" in data
    assert data["top5_acc"] >= 0.95
    assert "mrr" in data
    assert data["mrr"] >= 0.95
    assert "threshold_used" in data
    assert data["threshold_used"] > 0.0
    assert "tpir" in data
    assert "fpir" in data
    assert data["fpir"] <= 5.0


def test_get_evaluation_history():
    response = client.get("/api/v1/evaluation/history")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert data["total"] >= 1
    assert len(data["items"]) >= 1
    assert "model_version" in data["items"][0]
