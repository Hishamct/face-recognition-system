from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_get_training_runs():
    response = client.get("/api/v1/training/runs")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert data["total"] >= 1
    assert len(data["runs"]) >= 1

    first_run = data["runs"][0]
    assert "run_name" in first_run
    assert "status" in first_run
    assert first_run["status"] == "COMPLETED"
    assert first_run["best_val_acc"] is not None


def test_get_training_history():
    response = client.get("/api/v1/training/history")
    assert response.status_code == 200
    data = response.json()
    assert "run" in data
    assert "epochs" in data
    assert len(data["epochs"]) >= 1

    first_epoch = data["epochs"][0]
    assert "epoch" in first_epoch
    assert "train_loss" in first_epoch
    assert "train_acc" in first_epoch
    assert "val_loss" in first_epoch
    assert "val_acc" in first_epoch
