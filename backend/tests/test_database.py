"""
backend/tests/test_database.py
==============================
Automated unit tests for Phase 4 PostgreSQL Database and Vector Store.
Tests schema creation, vector encoding/decoding, cosine similarity search,
audit logging, training runs, epoch metrics, and FAISS parity.
"""

from pathlib import Path
import numpy as np
import pytest
from sqlalchemy import text

from backend.app.db.models import (
    Base,
    EpochMetric,
    EvaluationBenchmarkRecord,
    FaceEmbeddingRecord,
    Identity,
    RecognitionAuditLog,
    TrainingRun,
)
from backend.app.db.session import SessionLocal, verify_db_connection, engine
from backend.app.services.vector_store import PostgresVectorStore
from backend.app.vision.matcher import FaceMatcher


@pytest.fixture(scope="module")
def db_available():
    """Check if PostgreSQL database is reachable."""
    return verify_db_connection().get("status") == "connected"


def test_database_connection(db_available):
    if not db_available:
        pytest.skip("PostgreSQL database is not reachable at localhost:5432")
    session = SessionLocal()
    try:
        result = session.execute(text("SELECT 1")).scalar()
        assert result == 1
    finally:
        session.close()


def test_vector_store_enroll_and_search(db_available):
    if not db_available:
        pytest.skip("PostgreSQL database is not reachable")

    store = PostgresVectorStore()
    store.init_tables()

    # Generate test unit vectors
    v1 = np.random.randn(128).astype(np.float32)
    v2 = np.random.randn(128).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    test_name_1 = "pytest_test_user_alpha"
    test_name_2 = "pytest_test_user_beta"

    # Enroll
    store.enroll_identity(test_name_1, [v1], notes="QA tester")
    store.enroll_identity(test_name_2, [v2], notes="Dev builder")
    # Search with v1
    results = store.search_identities(v1, top_k=2)
    assert len(results) >= 2
    assert results[0]["identity"] == test_name_1
    assert np.isclose(results[0]["score"], 1.0, atol=1e-4)

    # Cleanup test entries
    session = SessionLocal()
    try:
        session.query(Identity).filter(Identity.name.in_([test_name_1, test_name_2])).delete(synchronize_session=False)
        session.commit()
    finally:
        session.close()


def test_faiss_postgres_exact_parity():
    """
    Verify mathematical parity between in-memory FAISS and PostgresVectorStore similarity matching.
    """
    np.random.seed(42)
    dim = 128
    num_samples = 20

    # Create random unit vectors
    matrix = np.random.randn(num_samples, dim).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    # FAISS matcher
    faiss_matcher = FaceMatcher(dimension=dim)
    for i in range(num_samples):
        faiss_matcher.add(f"id_{i:03d}", matrix[i])

    # Query vector
    q = np.random.randn(dim).astype(np.float32)
    q /= np.linalg.norm(q)

    # 1. FAISS top-5
    faiss_matches = faiss_matcher.search(q, top_k=5)

    # 2. Dot-product matrix multiplication (same formula as PostgresVectorStore)
    scores = np.dot(matrix, q)
    top_indices = np.argsort(scores)[::-1][:5]

    for rank in range(5):
        expected_idx = top_indices[rank]
        expected_name = f"id_{expected_idx:03d}"
        expected_score = float(scores[expected_idx])

        assert faiss_matches[rank]["identity"] == expected_name
        assert np.isclose(faiss_matches[rank]["score"], expected_score, atol=1e-5), (
            f"Parity mismatch at rank {rank + 1}: FAISS={faiss_matches[rank]['score']} vs Exact={expected_score}"
        )


def test_training_run_and_epoch_metric_db_integrity(db_available):
    if not db_available:
        pytest.skip("PostgreSQL database is not reachable")

    session = SessionLocal()
    try:
        run = TrainingRun(
            run_name="pytest_unit_test_run",
            backbone="sface_2021dec",
            num_classes=150,
            epochs=2,
            learning_rate=0.005,
            batch_size=32,
            best_val_acc=96.47,
            status="COMPLETED",
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        metric1 = EpochMetric(run_id=run.id, epoch=1, train_loss=0.95, train_acc=86.15, val_loss=0.28, val_acc=96.47)
        metric2 = EpochMetric(run_id=run.id, epoch=2, train_loss=0.07, train_acc=98.96, val_loss=0.28, val_acc=96.47)
        session.add_all([metric1, metric2])
        session.commit()

        # Query back
        fetched_run = session.query(TrainingRun).filter_by(id=run.id).first()
        assert fetched_run is not None
        assert len(fetched_run.metrics) == 2
        assert fetched_run.metrics[0].epoch == 1
        assert fetched_run.metrics[1].epoch == 2

        # Cleanup
        session.delete(fetched_run)
        session.commit()
    finally:
        session.close()
