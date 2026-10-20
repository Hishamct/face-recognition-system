"""
training/evaluate_recognition.py
=================================
Phase 3 - Evaluation Correction: Recognition Baseline Benchmark on Held-Out TEST Set.

Evaluates:
1. Closed-set 1:N Identification (Rank-1, Top-5 Accuracy, MRR)
   - Evaluated on distinct IDENTITIES using matcher.search_identities()
   - Ensures multiple exemplars per identity do not occupy multiple top-5 slots

2. Real Open-Set Unknown Rejection (Non-Mated Search)
   - Evaluated on 200 real human faces from un-enrolled identities (aligned_unknown_test)
   - Evaluated at both:
     * Frozen Pairwise Threshold (tau = 0.300)
     * Frozen Gallery Search Threshold (tau = 0.445, calibrated for gallery FPIR <= 5%)

3. Synthetic Vector Baseline Control
   - Random unit vectors on unit hypersphere (for comparison with real unknown face probes)

Outputs:
  reports/recognition_evaluation.json

Usage:
    python training/evaluate_recognition.py
"""

import argparse
import json
from pathlib import Path
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
    candidate_pool: int = 100,
) -> dict:
    """
    Evaluate 1:N Gallery Search on test images.
    Ranks distinct IDENTITIES, aggregating multiple exemplars per person.
    """
    id_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    total_queries = 0
    rank1_correct = 0
    top5_correct = 0
    rank_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "unranked": 0}
    reciprocal_ranks = []
    per_identity_results = {}

    for id_dir in tqdm(id_dirs, desc="Closed-Set 1:N Identification (Test Set)"):
        true_id = id_dir.name
        img_paths = sorted(id_dir.glob("*.jpg"))

        id_queries = 0
        id_correct = 0

        for img_p in img_paths:
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            query_vec = embedder.embed(img, normalize=True)
            # Retrieve ranked unique identities
            matches = matcher.search_identities(
                query_vec, top_k=top_k, candidate_pool=candidate_pool
            )
            if not matches:
                continue

            total_queries += 1
            id_queries += 1

            retrieved_ids = [m["identity"] for m in matches]

            if true_id in retrieved_ids:
                rank = retrieved_ids.index(true_id) + 1  # 1 to 5
                rank_distribution[rank] += 1
                reciprocal_ranks.append(1.0 / rank)
                top5_correct += 1
                if rank == 1:
                    rank1_correct += 1
                    id_correct += 1
            else:
                rank_distribution["unranked"] += 1
                reciprocal_ranks.append(0.0)

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
        "rank_distribution": rank_distribution,
        "per_identity": per_identity_results,
    }


def evaluate_open_set_identification(
    test_dir: Path,
    unknown_test_dir: Path,
    embedder: FaceEmbedder,
    matcher: FaceMatcher,
    threshold: float,
    num_synthetic_simulations: int = 200,
    seed: int = 42,
) -> dict:
    """
    Evaluate open-set identification & unknown rejection at frozen threshold tau.
    Evaluates:
    1. Known test queries (aligned_test, 624 images)
    2. Real unknown probes (aligned_unknown_test, 200 real human faces)
    3. Synthetic unit vectors (200 random vectors on unit sphere)
    """
    # 1. Known test probes
    known_imgs = sorted(list(test_dir.rglob("*.jpg")))
    known_total = 0
    known_correct_identified = 0
    known_false_rejects = 0
    known_misidentified = 0
    known_scores = []

    for img_p in known_imgs:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        vec = embedder.embed(img, normalize=True)
        res = matcher.identify(vec, threshold=threshold)
        true_id = img_p.parent.name
        known_total += 1
        known_scores.append(res["score"])

        if res["matched"]:
            if res["identity"] == true_id:
                known_correct_identified += 1
            else:
                known_misidentified += 1
        else:
            known_false_rejects += 1

    # 2. Real unknown human probes
    unknown_imgs = sorted(list(unknown_test_dir.rglob("*.jpg"))) if unknown_test_dir.exists() else []
    real_unk_total = 0
    real_unk_rejected = 0
    real_unk_false_accepts = 0
    real_unk_scores = []

    for img_p in unknown_imgs:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        vec = embedder.embed(img, normalize=True)
        res = matcher.identify(vec, threshold=threshold)
        real_unk_total += 1
        real_unk_scores.append(res["score"])

        if not res["matched"]:
            real_unk_rejected += 1
        else:
            real_unk_false_accepts += 1

    # 3. Synthetic orthogonal vector simulation (for comparison)
    synth_total = num_synthetic_simulations
    synth_rejected = 0
    synth_false_accepts = 0
    np_rng = np.random.RandomState(seed)

    for _ in range(synth_total):
        rand_vec = np_rng.randn(embedder.embedding_dimension).astype(np.float32)
        rand_vec /= np.linalg.norm(rand_vec)
        res = matcher.identify(rand_vec, threshold=threshold)
        if not res["matched"]:
            synth_rejected += 1
        else:
            synth_false_accepts += 1

    return {
        "decision_threshold_tau": threshold,
        "known_probes": {
            "total": known_total,
            "correctly_identified": known_correct_identified,
            "true_identification_rate_percent": (known_correct_identified / known_total * 100) if known_total else 0,
            "false_rejected": known_false_rejects,
            "false_rejection_rate_percent": (known_false_rejects / known_total * 100) if known_total else 0,
            "misidentified_as_other_enrolled": known_misidentified,
            "mean_score": float(np.mean(known_scores)) if known_scores else 0.0,
        },
        "real_unknown_human_probes": {
            "total": real_unk_total,
            "correctly_rejected": real_unk_rejected,
            "unknown_rejection_rate_percent": (real_unk_rejected / real_unk_total * 100) if real_unk_total else 0,
            "false_accepted_as_enrolled": real_unk_false_accepts,
            "false_positive_identification_rate_percent": (real_unk_false_accepts / real_unk_total * 100) if real_unk_total else 0,
            "mean_score": float(np.mean(real_unk_scores)) if real_unk_scores else 0.0,
        },
        "synthetic_random_vector_simulation": {
            "total": synth_total,
            "correctly_rejected": synth_rejected,
            "rejection_rate_percent": (synth_rejected / synth_total * 100) if synth_total else 0,
            "false_accepted": synth_false_accepts,
            "false_accept_rate_percent": (synth_false_accepts / synth_total * 100) if synth_total else 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 - Evaluate Recognition on Test Set")
    parser.add_argument("--test-dir", type=str, default="data/processed/aligned_test",
                        help="Path to aligned test dataset")
    parser.add_argument("--unknown-test-dir", type=str, default="data/processed/aligned_unknown_test",
                        help="Path to aligned real unknown test dataset")
    parser.add_argument("--model-path", type=str, default="models/face_recognition_sface_2021dec.onnx",
                        help="Path to SFace ONNX model")
    parser.add_argument("--db-dir", type=str, default="data/database",
                        help="Path to FAISS index database")
    parser.add_argument("--reports-dir", type=str, default="reports",
                        help="Reports output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    test_dir = project_root / args.test_dir
    unknown_test_dir = project_root / args.unknown_test_dir
    model_path = project_root / args.model_path
    db_dir = project_root / args.db_dir
    reports_dir = project_root / args.reports_dir

    print("=" * 65)
    print("  Phase 3 - Recognition Benchmark Evaluation on Held-Out TEST Set")
    print("=" * 65)
    print(f"  Test Dataset (Known)    : {test_dir}")
    print(f"  Test Dataset (Unknown)  : {unknown_test_dir}")
    print(f"  Embedding Model         : {model_path}")
    print(f"  FAISS Gallery Index     : {db_dir / 'faiss_index.bin'}")
    print()

    # Load calibration thresholds
    calib_json = reports_dir / "threshold_calibration.json"
    if calib_json.exists():
        with open(calib_json, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        tau_pairwise = calib_data["pairwise_1_to_1_verification"]["operational_threshold_far1pct"]
        tau_gallery = calib_data["gallery_level_1_to_n_search"]["tau_gallery_fpir5pct"]
        print(f"  Loaded Frozen 1:1 Pairwise Threshold tau       : {tau_pairwise:.3f}")
        print(f"  Loaded Frozen 1:N Gallery Search Threshold tau : {tau_gallery:.3f}")
    else:
        tau_pairwise = 0.300
        tau_gallery = 0.445
        print(f"  Using Default Thresholds: pairwise={tau_pairwise:.3f}, gallery={tau_gallery:.3f}")

    embedder = FaceEmbedder(model_path=model_path)
    matcher = FaceMatcher(dimension=embedder.embedding_dimension)
    matcher.load(db_dir / "faiss_index.bin", db_dir / "gallery_metadata.json")
    print(f"  Loaded FAISS Gallery Index with {matcher.ntotal} enrolled vectors across {len(set(matcher.identities))} identities")

    # 1. Closed-set 1:N Identification (deduplicated by identity)
    print("\n[Step 1: Closed-Set 1:N Identification Benchmark]")
    closed_set_results = evaluate_closed_set_identification(test_dir, embedder, matcher, top_k=5)

    # 2. Open-set Rejection at Pairwise Threshold (tau_pairwise = 0.300)
    print(f"\n[Step 2A: Open-Set Evaluation @ 1:1 Pairwise Threshold tau = {tau_pairwise:.3f}]")
    open_set_pairwise = evaluate_open_set_identification(
        test_dir, unknown_test_dir, embedder, matcher, threshold=tau_pairwise
    )

    # 3. Open-set Rejection at Gallery Search Threshold (tau_gallery = 0.445)
    print(f"\n[Step 2B: Open-Set Evaluation @ 1:N Gallery Search Threshold tau = {tau_gallery:.3f}]")
    open_set_gallery = evaluate_open_set_identification(
        test_dir, unknown_test_dir, embedder, matcher, threshold=tau_gallery
    )

    # Compile final evaluation report
    report = {
        "model": "face_recognition_sface_2021dec",
        "embedding_dimension": embedder.embedding_dimension,
        "enrolled_gallery_size": matcher.ntotal,
        "enrolled_identities_count": len(set(matcher.identities)),
        "closed_set_identification": closed_set_results,
        "open_set_evaluation_pairwise_tau_0_300": open_set_pairwise,
        "open_set_evaluation_gallery_tau_0_445": open_set_gallery,
    }

    eval_out = reports_dir / "recognition_evaluation.json"
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print("  TEST SET RECOGNITION BENCHMARK RESULTS:")
    print("=" * 65)
    print("  [Closed-Set 1:N Identification (Ranked Identities)]")
    print(f"    Total Test Queries     : {closed_set_results['total_test_queries']}")
    print(f"    Rank-1 Accuracy        : {closed_set_results['rank1_accuracy_percent']:.2f}% ({closed_set_results['rank1_correct']}/{closed_set_results['total_test_queries']})")
    print(f"    Top-5 Accuracy         : {closed_set_results['top5_accuracy_percent']:.2f}% ({closed_set_results['top5_correct']}/{closed_set_results['total_test_queries']})")
    print(f"    Mean Reciprocal Rank   : {closed_set_results['mrr']:.4f}")
    print(f"    Rank Distribution      : {closed_set_results['rank_distribution']}")
    print()
    print(f"  [Open-Set Real Human Probes @ Calibrated Gallery tau = {tau_gallery:.3f}]")
    print(f"    Known True Identification Rate (TPIR)  : {open_set_gallery['known_probes']['true_identification_rate_percent']:.2f}% ({open_set_gallery['known_probes']['correctly_identified']}/{open_set_gallery['known_probes']['total']})")
    print(f"    Real Unknown Rejection Rate            : {open_set_gallery['real_unknown_human_probes']['unknown_rejection_rate_percent']:.2f}% ({open_set_gallery['real_unknown_human_probes']['correctly_rejected']}/{open_set_gallery['real_unknown_human_probes']['total']})")
    print(f"    Real Unknown False Positive ID (FPIR)  : {open_set_gallery['real_unknown_human_probes']['false_positive_identification_rate_percent']:.2f}% ({open_set_gallery['real_unknown_human_probes']['false_accepted_as_enrolled']}/{open_set_gallery['real_unknown_human_probes']['total']})")
    print()
    print(f"  [Comparison with 1:1 Pairwise Threshold tau = {tau_pairwise:.3f}]")
    print(f"    Known True Identification Rate (TPIR)  : {open_set_pairwise['known_probes']['true_identification_rate_percent']:.2f}%")
    print(f"    Real Unknown Rejection Rate            : {open_set_pairwise['real_unknown_human_probes']['unknown_rejection_rate_percent']:.2f}%")
    print(f"    Real Unknown False Positive ID (FPIR)  : {open_set_pairwise['real_unknown_human_probes']['false_positive_identification_rate_percent']:.2f}%")
    print()
    print(f"  [Synthetic Orthogonal Vectors vs Real Face Vectors]")
    print(f"    Synthetic Vector Rejection @ tau={tau_pairwise:.3f} : {open_set_pairwise['synthetic_random_vector_simulation']['rejection_rate_percent']:.2f}%")
    print(f"    Real Human Rejection @ tau={tau_pairwise:.3f}       : {open_set_pairwise['real_unknown_human_probes']['unknown_rejection_rate_percent']:.2f}%")
    print(f"    Synthetic Vector Rejection @ tau={tau_gallery:.3f}  : {open_set_gallery['synthetic_random_vector_simulation']['rejection_rate_percent']:.2f}%")
    print(f"    Real Human Rejection @ tau={tau_gallery:.3f}        : {open_set_gallery['real_unknown_human_probes']['unknown_rejection_rate_percent']:.2f}%")
    print(f"\n  Full Evaluation Report Saved -> {eval_out}")
    print("=" * 65)


if __name__ == "__main__":
    main()
