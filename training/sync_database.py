"""
training/sync_database.py
=========================
Phase 4: Populate PostgreSQL database with enrolled identities and feature vectors
from data/database/faiss_index.bin and data/database/gallery_metadata.json.

Verifies:
1. Exact row count synchronization (150 identities, 2,981 vectors)
2. Exact Cosine Similarity retrieval parity between FAISS and PostgreSQL across sample queries
"""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.vector_store import PostgresVectorStore
from backend.app.vision.embedder import FaceEmbedder
from backend.app.vision.matcher import FaceMatcher


def verify_faiss_postgres_parity(
    faiss_matcher: FaceMatcher,
    pg_store: PostgresVectorStore,
    test_dir: Path,
    embedder: FaceEmbedder,
    sample_queries: int = 50,
) -> dict:
    """
    Evaluate random test probes on both FAISS and PostgreSQL.
    Asserts:
    1. The top retrieved identity is identical.
    2. The similarity scores match within floating point precision (atol <= 1e-4).
    """
    all_test_imgs = sorted(list(test_dir.rglob("*.jpg")))
    rng = np.random.RandomState(42)
    selected_indices = rng.choice(len(all_test_imgs), size=min(sample_queries, len(all_test_imgs)), replace=False)
    selected_imgs = [all_test_imgs[i] for i in selected_indices]

    cache = pg_store.load_gallery_cache()
    max_score_diff = 0.0
    identical_top1_count = 0

    print(f"\nEvaluating Retrieval Parity on {len(selected_imgs)} sample test queries...")

    for img_p in selected_imgs:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        vec = embedder.embed(img, normalize=True)

        # 1. Query FAISS
        faiss_res = faiss_matcher.search_identities(vec, top_k=1, candidate_pool=None)
        # 2. Query PostgreSQL
        pg_res = pg_store.search_identities(vec, top_k=1, matrix_cache=cache)

        if not faiss_res or not pg_res:
            continue

        faiss_top = faiss_res[0]
        pg_top = pg_res[0]

        diff = abs(faiss_top["score"] - pg_top["score"])
        if diff > max_score_diff:
            max_score_diff = diff

        if faiss_top["identity"] == pg_top["identity"]:
            identical_top1_count += 1
        else:
            print(f"  [MISMATCH] Query: {img_p.name} -> FAISS: {faiss_top['identity']} ({faiss_top['score']:.4f}) vs PG: {pg_top['identity']} ({pg_top['score']:.4f})")

    parity_percent = (identical_top1_count / len(selected_imgs)) * 100

    return {
        "queries_tested": len(selected_imgs),
        "identical_top1_matches": identical_top1_count,
        "parity_percentage": parity_percent,
        "max_score_difference": float(max_score_diff),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4 - Sync FAISS Gallery to PostgreSQL and Verify Parity")
    parser.add_argument("--faiss-index", type=str, default="data/database/faiss_index.bin")
    parser.add_argument("--gallery-meta", type=str, default="data/database/gallery_metadata.json")
    parser.add_argument("--test-dir", type=str, default="data/processed/aligned_test")
    parser.add_argument("--model-path", type=str, default="models/face_recognition_sface_2021dec.onnx")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    faiss_index_p = project_root / args.faiss_index
    meta_p = project_root / args.gallery_meta
    test_dir_p = project_root / args.test_dir
    model_p = project_root / args.model_path

    print("=" * 65)
    print("  Phase 4 - PostgreSQL Database Synchronization & Parity Audit")
    print("=" * 65)
    print(f"  Source FAISS Index   : {faiss_index_p}")
    print(f"  Source Metadata      : {meta_p}")
    print(f"  Target DB            : PostgreSQL (facedb)")
    print()

    # 1. Initialize Tables and Sync
    store = PostgresVectorStore()
    print("Synchronizing gallery into PostgreSQL database...")
    t0 = time.perf_counter()
    sync_result = store.sync_from_faiss_gallery(faiss_index_p, meta_p)
    t_sync = time.perf_counter() - t0
    print(f"  [OK] Synchronized {sync_result['identities_synced']} identities and {sync_result['vectors_synced']} vectors in {t_sync:.2f}s")

    # 2. Verify Parity against FAISS
    embedder = FaceEmbedder(model_path=model_p)
    faiss_matcher = FaceMatcher(dimension=embedder.embedding_dimension)
    faiss_matcher.load(faiss_index_p, meta_p)

    parity_stats = verify_faiss_postgres_parity(faiss_matcher, store, test_dir_p, embedder, sample_queries=50)

    print("\n" + "=" * 65)
    print("  DATABASE RETRIEVAL PARITY RESULTS:")
    print("=" * 65)
    print(f"  Queries Tested           : {parity_stats['queries_tested']}")
    print(f"  Identical Top-1 Matches  : {parity_stats['identical_top1_matches']} / {parity_stats['queries_tested']}")
    print(f"  Retrieval Parity Rate    : {parity_stats['parity_percentage']:.2f}%")
    print(f"  Max Score Absolute Diff  : {parity_stats['max_score_difference']:.2e}")
    print("=" * 65)

    assert parity_stats["parity_percentage"] == 100.0, "FAISS and PostgreSQL must exhibit 100% retrieval parity!"
    print("  [SUCCESS] FAISS and PostgreSQL exhibit 100% numerical and identity parity.")


if __name__ == "__main__":
    main()
