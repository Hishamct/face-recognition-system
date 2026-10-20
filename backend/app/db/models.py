"""
backend/app/db/models.py
========================
SQLAlchemy database models for face recognition system:
- Identity: enrolled person records
- FaceEmbeddingRecord: enrolled exemplar and prototype feature vectors
- RecognitionAuditLog: real-time recognition query audit log
- TrainingRun: model fine-tuning experiments
- EpochMetric: epoch-by-epoch loss & accuracy history
- EvaluationBenchmarkRecord: multi-metric test set evaluations
"""

from datetime import datetime
from typing import Optional
import numpy as np

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

try:
    from backend.app.db.session import Base
except ImportError:
    from app.db.session import Base


class Identity(Base):
    """Enrolled person identity in the recognition platform."""
    __tablename__ = "identities"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    embeddings = relationship("FaceEmbeddingRecord", back_populates="identity", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Identity(id={self.id}, name='{self.name}', active={self.is_active})>"


class FaceEmbeddingRecord(Base):
    """Enrolled embedding vector for an identity (exemplar or centroid prototype)."""
    __tablename__ = "face_embeddings"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    identity_id = Column(Integer, ForeignKey("identities.id", ondelete="CASCADE"), nullable=False, index=True)
    exemplar_type = Column(String(20), default="exemplar", nullable=False)  # 'exemplar' or 'prototype'
    embedding_dim = Column(Integer, default=128, nullable=False)
    embedding_data = Column(LargeBinary, nullable=False)  # Float32 buffer (512 bytes for 128-D)
    norm = Column(Float, default=1.0, nullable=False)
    source_image = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    identity = relationship("Identity", back_populates="embeddings")

    def get_vector(self) -> np.ndarray:
        """Decode binary buffer into a 128-D float32 numpy vector."""
        return np.frombuffer(self.embedding_data, dtype=np.float32)

    def set_vector(self, vec: np.ndarray) -> None:
        """Store float32 numpy vector as binary buffer."""
        v = vec.astype(np.float32).flatten()
        self.embedding_dim = len(v)
        self.embedding_data = v.tobytes()
        self.norm = float(np.linalg.norm(v))

    def __repr__(self) -> str:
        return f"<FaceEmbeddingRecord(id={self.id}, identity_id={self.identity_id}, type='{self.exemplar_type}')>"


class RecognitionAuditLog(Base):
    """Audit log of face recognition queries, matching decisions, and latencies."""
    __tablename__ = "recognition_audit_logs"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    query_timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True, nullable=False)
    matched_identity = Column(String(100), nullable=True)
    similarity_score = Column(Float, nullable=False)
    threshold_used = Column(Float, nullable=False)
    decision = Column(String(50), nullable=False)  # 'RECOGNIZED', 'UNKNOWN', 'MISIDENTIFIED'
    latency_ms = Column(Float, nullable=False)
    client_ip = Column(String(50), nullable=True)

    def __repr__(self) -> str:
        return f"<RecognitionAuditLog(id={self.id}, match='{self.matched_identity}', score={self.similarity_score:.3f}, decision='{self.decision}')>"


class TrainingRun(Base):
    """Record of a model training or fine-tuning experiment."""
    __tablename__ = "training_runs"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    run_name = Column(String(100), unique=True, index=True, nullable=False)
    backbone = Column(String(100), default="sface_2021dec", nullable=False)
    num_classes = Column(Integer, default=150, nullable=False)
    epochs = Column(Integer, default=20, nullable=False)
    learning_rate = Column(Float, default=0.001, nullable=False)
    batch_size = Column(Integer, default=32, nullable=False)
    best_val_acc = Column(Float, nullable=True)
    best_val_loss = Column(Float, nullable=True)
    status = Column(String(50), default="RUNNING", nullable=False)  # 'RUNNING', 'COMPLETED', 'FAILED'
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    metrics = relationship("EpochMetric", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<TrainingRun(id={self.id}, name='{self.run_name}', status='{self.status}')>"


class EpochMetric(Base):
    """Epoch-by-epoch loss and accuracy metrics."""
    __tablename__ = "epoch_metrics"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float, nullable=False)
    train_acc = Column(Float, nullable=False)
    val_loss = Column(Float, nullable=False)
    val_acc = Column(Float, nullable=False)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    run = relationship("TrainingRun", back_populates="metrics")

    def __repr__(self) -> str:
        return f"<EpochMetric(run_id={self.run_id}, epoch={self.epoch}, train_loss={self.train_loss:.4f}, val_acc={self.val_acc:.2f}%)>"


class EvaluationBenchmarkRecord(Base):
    """Summary of full test-set evaluation benchmarks."""
    __tablename__ = "evaluation_benchmarks"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    evaluation_name = Column(String(100), nullable=False)
    test_split = Column(String(50), default="aligned_test", nullable=False)
    rank1_acc = Column(Float, nullable=False)
    top5_acc = Column(Float, nullable=False)
    mrr = Column(Float, nullable=False)
    tpir = Column(Float, nullable=False)
    fpir = Column(Float, nullable=False)
    num_known_probes = Column(Integer, default=624, nullable=False)
    num_unknown_probes = Column(Integer, default=200, nullable=False)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<EvaluationBenchmarkRecord(name='{self.evaluation_name}', rank1={self.rank1_acc:.2f}%, mrr={self.mrr:.4f})>"
