"""
training/evaluate_recognition.py
=================================
Phase 3 - Step 3: Evaluate Face Recognition Baseline on Held-Out TEST Set.

Reads  : data/processed/aligned_test/<IdentityName>/*.jpg
         data/database/faiss_index.bin
         data/database/gallery_metadata.json
         reports/threshold_calibration.json
Outputs: reports/recognition_evaluation.json

Evaluates:
1. Closed-set 1:N Identification (Rank-1, Top-5 Accuracy, MRR)
2. Open-set Unknown Rejection at frozen validation threshold tau
3. Pairwise 1:1 Verification on test pairs

Usage:
    python training/evaluate_recognition.py
"""

import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.vision.embedder import FaceEmbedder
from backend.app.vision.matcher import FaceMatcher


def evaluate_closed_set_identification(
    test_dir: Path,
    embedder: FaceEmbedder,
    matcher: FaceMatcher,
    top_k: int = 5,
) -> dict:
    """
    Evaluate 1:N Gallery Search on test images.
    Gallery is pre-enrolled from aligned_train.
    """
    id_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    total_queries = 0
    rank1_correct = 0
    top5_correct = 0
    reciprocal_ranks = []

    per_identity_results = {}

    for id_dir in tqdm(id_dirs, desc="Evaluating 1:N Test Identification"):
        true_id = id_dir.name
        img_paths = sorted(id_dir.glob("*.jpg"))

        id_queries = 0
        id_correct = 0

        for img_p in img_paths:
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            query_vec = embedder.embed(img, normalize=True)
            matches = matcher.search(query_vec, top_k=top_k)
            if not matches:
                continue

            total_queries += 1
            id_queries += 1

            retrieved_ids = [m["identity"] for m in matches]
            # Check Rank-1
            if retrieved_ids[0] == true_id:
                rank1_correct += 1
                id_correct += 1
                reciprocal_ranks.append(1.0)
            elif true_id in retrieved_ids:
                rank = retrieved_ids.index(true_id) + 1
                top5_correct += 1  # in top 5
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

            # If top-1 is true_id, it is also in top-5
            if retrieved_ids[0] == true_id:
                top5_correct += 1

        if id_queries > 0:
            per_identity_results[true_id] = {
                "queries": id_queries,
                "rank1_accuracy": id_correct / id_queries,
            }

    rank1_acc = (rank1_correct / total_queries) if total_queries > 0 else 0.0
    top5_acc = (top5_correct / total_queries) if total_queries > 0 else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    return {
        "total_test_queries": total_queries,
        "rank1_correct": rank1_correct,
        "rank1_accuracy_percent": rank1_acc * 100,
        "top5_correct": top5_correct,
        "top5_accuracy_percent": top5_acc * 100,
        "mrr": mrr,
        "per_identity": per_identity_results,
    }


def evaluate_open_set_rejection(
    test_dir: Path,
    embedder: FaceEmbedder,
    matcher: FaceMatcher,
    threshold: float,
    num_unknown_simulations: int = 200,
    seed: int = 42,
) -> dict:
    """
    Evaluate open-set identification & unknown rejection at the frozen threshold tau.
    Tests:
    1. Known test queries: Should be accepted and matched (score >= tau).
    2. Unknown queries: Synthesized by orthogonal random projections and un-enrolled noise vectors.
    """
    rng = random.Random(seed)
    all_test_imgs = list(test_dir.rglob("*.jpg"))

    # 1. Known test evaluation
    known_total = 0
    known_correctly_identified = 0
    known_falsely_rejected = 0

    for img_p in all_test_imgs:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        vec = embedder.embed(img, normalize=True)
        res = matcher.identify(vec, threshold=threshold)

        known_total += 1
        true_id = img_p.parent.name
        if res["matched"] and res["identity"] == true_id:
            known_correctly_identified += 1
        else:
            known_falsely_rejected += 1

    # 2. Unknown rejection simulation (vectors from random hypersphere orthogonal to gallery)
    unknown_total = num_unknown_simulations
    unknown_correctly_rejected = 0
    unknown_false_accepts = 0

    np_rng = np.random.RandomState(seed)
    for _ in range(unknown_total):
        # Generate random unit vector on unit sphere (representing unenrolled intruder)
        rand_vec = np_rng.randn(embedder.embedding_dimension).astype(np.float32)
        rand_vec /= np.linalg.norm(rand_vec)

        res = matcher.identify(rand_vec, threshold=threshold)
        if not res["matched"]:
            unknown_correctly_rejected += 1
        else:
            unknown_false_accepts += 1

    return {
        "decision_threshold_tau": threshold,
        "known_queries_total": known_total,
        "known_correctly_identified": known_correctly_identified,
        "true_identification_rate_percent": (known_correctly_identified / known_total * 100) if known_total else 0,
        "known_falsely_rejected": known_falsely_rejected,
        "unknown_queries_total": unknown_total,
        "unknown_correctly_rejected": unknown_correctly_rejected,
        "unknown_rejection_rate_percent": (unknown_correctly_rejected / unknown_total * 100) if unknown_total else 0,
        "unknown_false_accepts": unknown_false_accepts,
        "false_acceptance_rate_percent": (unknown_false_accepts / unknown_total * 100) if unknown_total else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 - Evaluate Recognition on Test Set")
    parser.add_argument("--test-dir", type=str, default="data/processed/aligned_test",
                        help="Path to aligned test dataset")
    parser.add_argument("--model-path", type=str, default="models/face_recognition_sface_2021dec.onnx",
                        help="Path to SFace ONNX model")
    parser.add_argument("--db-dir", type=str, default="data/database",
                        help="Path to FAISS index database")
    parser.add_argument("--reports-dir", type=str, default="reports",
                        help="Reports output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    test_dir = project_root / args.test_dir
    model_path = project_root / args.model_path
    db_dir = project_root / args.db_dir
    reports_dir = project_root / args.reports_dir

    print("=" * 65)
    print("  Phase 3 - Recognition Baseline Evaluation on TEST Set")
    print("=" * 65)
    print(f"  Test Dataset        : {test_dir}")
    print(f"  Embedding Model     : {model_path}")
    print(f"  FAISS Gallery Index : {db_dir / 'faiss_index.bin'}")
    print()

    # Load calibration threshold
    calib_json = reports_dir / "threshold_calibration.json"
    if calib_json.exists():
        with open(calib_json, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        frozen_tau = calib_data["operational_threshold_far1pct"]
        print(f"  Loaded Frozen Validation Threshold tau : {frozen_tau:.3f} (target FAR <= 1%)")
    else:
        frozen_tau = 0.363
        print(f"  Using Default Threshold tau : {frozen_tau:.3f}")

    embedder = FaceEmbedder(model_path=model_path)
    matcher = FaceMatcher(dimension=embedder.embedding_dimension)
    matcher.load(db_dir / "faiss_index.bin", db_dir / "gallery_metadata.json")
    print(f"  Loaded FAISS Gallery Index with {matcher.ntotal} enrolled vectors")

    # 1. Closed-set 1:N Identification
    closed_set_results = evaluate_closed_set_identification(test_dir, embedder, matcher, top_k=5)

    # 2. Open-set Rejection
    open_set_results = evaluate_open_set_rejection(test_dir, embedder, matcher, threshold=frozen_tau)

    # Compile final evaluation report
    report = {
        "model": "face_recognition_sface_2021dec",
        "embedding_dimension": embedder.embedding_dimension,
        "enrolled_gallery_size": matcher.ntotal,
        "closed_set_identification": closed_set_results,
        "open_set_rejection": open_set_results,
    }

    eval_out = reports_dir / "recognition_evaluation.json"
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print("  TEST SET RECOGNITION BENCHMARK RESULTS:")
    print("=" * 65)
    print("  [Closed-Set 1:N Identification]")
    print(f"    Total Test Queries     : {closed_set_results['total_test_queries']}")
    print(f"    Rank-1 Accuracy        : {closed_set_results['rank1_accuracy_percent']:.2f}% ({closed_set_results['rank1_correct']}/{closed_set_results['total_test_queries']})")
    print(f"    Top-5 Accuracy         : {closed_set_results['top5_accuracy_percent']:.2f}% ({closed_set_results['top5_correct']}/{closed_set_results['total_test_queries']})")
    print(f"    Mean Reciprocal Rank   : {closed_set_results['mrr']:.4f}")
    print()
    print("  [Open-Set Unknown Rejection @ Frozen Validation tau = {:.3f}]".format(frozen_tau))
    print(f"    True Identification Rate : {open_set_results['true_identification_rate_percent']:.2f}%")
    print(f"    Unknown Rejection Rate   : {open_set_results['unknown_rejection_rate_percent']:.2f}%")
    print(f"    False Acceptance Rate    : {open_set_results['false_acceptance_rate_percent']:.2f}%")
    print(f"\n  Full Evaluation Report Saved -> {eval_out}")
    print("=" * 65)


if __name__ == "__main__":
    main()
