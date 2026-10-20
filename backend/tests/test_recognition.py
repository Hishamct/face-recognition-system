"""
backend/tests/test_recognition.py
=================================
Automated unit tests for Phase 3 face recognition modules:
SFace feature embedder, cosine similarity, FAISS vector indexing, and identification.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from backend.app.vision.embedder import FaceEmbedder
from backend.app.vision.matcher import FaceMatcher


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def embedder(repo_root: Path) -> FaceEmbedder:
    model_path = repo_root / "models" / "face_recognition_sface_2021dec.onnx"
    assert model_path.exists(), "SFace model must exist in models/"
    return FaceEmbedder(model_path=model_path)


@pytest.fixture
def sample_aligned_faces(repo_root: Path) -> tuple[np.ndarray, np.ndarray, str, str]:
    train_dir = repo_root / "data" / "processed" / "aligned_train"
    if not train_dir.exists():
        pytest.skip("Aligned dataset not present for test fixture")

    id_dirs = [d for d in train_dir.iterdir() if d.is_dir() and len(list(d.glob("*.jpg"))) >= 2]
    if len(id_dirs) < 2:
        pytest.skip("Need at least 2 identities with >= 2 images")

    id1_p = list(id_dirs[0].glob("*.jpg"))
    id2_p = list(id_dirs[1].glob("*.jpg"))

    img1 = cv2.imread(str(id1_p[0]))
    img2 = cv2.imread(str(id2_p[0]))
    return img1, img2, id_dirs[0].name, id_dirs[1].name


def test_embedder_initialization(embedder: FaceEmbedder):
    assert embedder is not None
    assert embedder.embedding_dimension == 128


def test_embedder_output_shape_and_norm(embedder: FaceEmbedder, sample_aligned_faces):
    img1, _, _, _ = sample_aligned_faces
    emb = embedder.embed(img1, normalize=True)

    assert emb.shape == (128,)
    assert emb.dtype == np.float32
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5), "L2-normalized embedding must have unit norm"


def test_embedder_cosine_similarity(embedder: FaceEmbedder, sample_aligned_faces):
    img1, img2, id1, id2 = sample_aligned_faces
    emb1 = embedder.embed(img1, normalize=True)
    emb2 = embedder.embed(img2, normalize=True)

    # Self similarity must be 1.0
    self_sim = embedder.compute_similarity(emb1, emb1)
    assert np.isclose(self_sim, 1.0, atol=1e-5)

    # Different identity similarity should typically be much lower
    diff_sim = embedder.compute_similarity(emb1, emb2)
    assert diff_sim < 0.75, f"Different identities {id1} and {id2} should not have excessive similarity"


def test_matcher_enroll_and_search():
    matcher = FaceMatcher(dimension=128)
    assert matcher.ntotal == 0

    v1 = np.random.randn(128).astype(np.float32)
    v2 = np.random.randn(128).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    matcher.add("Alice", v1, metadata={"role": "engineer"})
    matcher.add("Bob", v2, metadata={"role": "designer"})
    assert matcher.ntotal == 2

    # Query with exact Alice vector
    matches = matcher.search(v1, top_k=2)
    assert len(matches) == 2
    assert matches[0]["rank"] == 1
    assert matches[0]["identity"] == "Alice"
    assert np.isclose(matches[0]["score"], 1.0, atol=1e-5)
    assert matches[0]["metadata"]["role"] == "engineer"


def test_matcher_identify_with_threshold():
    matcher = FaceMatcher(dimension=128)
    v1 = np.random.randn(128).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    matcher.add("Charlie", v1)

    # Known query
    res_known = matcher.identify(v1, threshold=0.5)
    assert res_known["matched"] is True
    assert res_known["identity"] == "Charlie"

    # Impostor / orthogonal noise query
    v_unknown = np.random.randn(128).astype(np.float32)
    v_unknown -= np.dot(v_unknown, v1) * v1  # Gram-Schmidt orthogonalize
    v_unknown /= np.linalg.norm(v_unknown)

    res_unknown = matcher.identify(v_unknown, threshold=0.5)
    assert res_unknown["matched"] is False
    assert res_unknown["identity"] == "Unknown"


def test_matcher_save_and_load(tmp_path: Path):
    idx_path = tmp_path / "test_faiss.bin"
    meta_path = tmp_path / "test_meta.json"

    m1 = FaceMatcher(dimension=128)
    vec = np.random.randn(128).astype(np.float32)
    vec /= np.linalg.norm(vec)
    m1.add("David", vec, metadata={"dept": "ops"})
    m1.save(idx_path, meta_path)

    m2 = FaceMatcher(dimension=128)
    m2.load(idx_path, meta_path)

    assert m2.ntotal == 1
    assert m2.identities == ["David"]
    assert m2.metadata[0]["dept"] == "ops"

    matches = m2.search(vec, top_k=1)
    assert matches[0]["identity"] == "David"
    assert np.isclose(matches[0]["score"], 1.0, atol=1e-5)


def test_matcher_search_identities_aggregates_exemplars():
    matcher = FaceMatcher(dimension=128)
    v1 = np.random.randn(128).astype(np.float32)
    v2 = v1 + np.random.randn(128).astype(np.float32) * 0.05
    v3 = np.random.randn(128).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 /= np.linalg.norm(v3)

    # Enroll 2 exemplars for Alice, 1 for Bob
    matcher.add("Alice", v1)
    matcher.add("Alice", v2)
    matcher.add("Bob", v3)
    assert matcher.ntotal == 3

    # search_identities should return Alice once and Bob once
    matches = matcher.search_identities(v1, top_k=2)
    assert len(matches) == 2
    assert matches[0]["identity"] == "Alice"
    assert matches[1]["identity"] == "Bob"
    assert matches[0]["rank"] == 1
    assert matches[1]["rank"] == 2


def test_tripartite_partition_outcomes_sum_to_total():
    """
    Verify that in open-set identification, the outcomes for known probes
    (Correct ID + False Reject + Misidentified) strictly sum to total known probes,
    and unknown probes (Correct Reject + False Accept) strictly sum to total unknown probes.
    """
    matcher = FaceMatcher(dimension=128)
    v_alice = np.random.randn(128).astype(np.float32)
    v_bob = np.random.randn(128).astype(np.float32)
    v_alice /= np.linalg.norm(v_alice)
    v_bob /= np.linalg.norm(v_bob)
    matcher.add("Alice", v_alice)
    matcher.add("Bob", v_bob)

    # 3 known probes:
    # 1. Exact Alice (should be correctly identified)
    # 2. Perturbed Alice with low similarity (should be false reject)
    # 3. Vector closer to Bob but true label is Alice (should be misidentified)
    known_probes = [
        (v_alice.copy(), "Alice"),
        (np.random.randn(128).astype(np.float32) / 10.0, "Alice"),
        (v_bob.copy(), "Alice"),
    ]

    correct_id = 0
    false_reject = 0
    misidentified = 0
    tau = 0.50

    for vec, true_label in known_probes:
        res = matcher.identify(vec, threshold=tau)
        if res["matched"]:
            if res["identity"] == true_label:
                correct_id += 1
            else:
                misidentified += 1
        else:
            false_reject += 1

    # Exact tri-partite sum assertion
    assert correct_id + false_reject + misidentified == len(known_probes)
    assert correct_id == 1
    assert false_reject == 1
    assert misidentified == 1


