"""
backend/tests/test_classifier.py
================================
Automated unit tests for Phase 4 PyTorch supervised classification head:
IdentityClassifier architecture, forward pass, top-k prediction, checkpoint persistence,
and real model weights validation.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from backend.app.vision.classifier import IdentityClassifier


def test_classifier_initialization():
    class_map = {f"Person_{i}": i for i in range(150)}
    model = IdentityClassifier(embedding_dim=128, num_classes=150, dropout_p=0.2, class_to_idx=class_map)

    assert model.embedding_dim == 128
    assert model.num_classes == 150
    assert len(model.class_to_idx) == 150
    assert len(model.idx_to_class) == 150


def test_classifier_forward_pass():
    model = IdentityClassifier(embedding_dim=128, num_classes=150)
    batch_size = 8
    dummy_input = torch.randn(batch_size, 128)
    logits = model(dummy_input)

    assert logits.shape == (batch_size, 150)
    assert not torch.isnan(logits).any()


def test_classifier_predict_embedding():
    class_map = {"Alice": 0, "Bob": 1, "Charlie": 2}
    model = IdentityClassifier(embedding_dim=128, num_classes=3, class_to_idx=class_map)

    dummy_vec = np.random.randn(128).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)

    top1_name, top1_prob, top_k = model.predict_embedding(dummy_vec, top_k=3)

    assert top1_name in class_map
    assert 0.0 <= top1_prob <= 1.0
    assert len(top_k) == 3
    assert top_k[0]["rank"] == 1
    assert top_k[1]["rank"] == 2
    assert top_k[2]["rank"] == 3
    assert top_k[0]["probability"] >= top_k[1]["probability"]
    assert top_k[1]["probability"] >= top_k[2]["probability"]


def test_classifier_save_and_load(tmp_path: Path):
    ckpt_path = tmp_path / "test_head.pt"
    class_map = {"Alpha": 0, "Beta": 1}
    m1 = IdentityClassifier(embedding_dim=128, num_classes=2, class_to_idx=class_map)
    m1.save(ckpt_path)

    assert ckpt_path.exists()
    m2 = IdentityClassifier.load(ckpt_path, device="cpu")

    assert m2.embedding_dim == 128
    assert m2.num_classes == 2
    assert m2.class_to_idx == class_map

    dummy_vec = np.random.randn(128).astype(np.float32)
    m1.eval()
    t1 = m1(torch.from_numpy(dummy_vec).unsqueeze(0))
    t2 = m2(torch.from_numpy(dummy_vec).unsqueeze(0))
    assert torch.allclose(t1, t2, atol=1e-5)


def test_trained_checkpoint_exists_and_valid():
    ckpt_path = Path("models/classifier_head.pt")
    if not ckpt_path.exists():
        pytest.skip("models/classifier_head.pt not yet generated")

    model = IdentityClassifier.load(ckpt_path, device="cpu")
    assert model.embedding_dim == 128
    assert model.num_classes == 150
    assert len(model.class_to_idx) == 150

    # Test probe prediction
    vec = np.random.randn(128).astype(np.float32)
    vec /= np.linalg.norm(vec)
    name, prob, top5 = model.predict_embedding(vec, top_k=5)

    assert isinstance(name, str)
    assert 0.0 <= prob <= 1.0
    assert len(top5) == 5
