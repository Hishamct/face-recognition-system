"""
training/calibrate_threshold.py
=================================
Phase 3 - Evaluation Correction: Dual-Level Decision Threshold Calibration on VALIDATION Set.

Distinguishes between two fundamental biometric operational modes:
1. Pairwise 1:1 Verification (Evaluated on genuine vs impostor pairs in aligned_val)
   - Genuine pairs: same identity, different validation images
   - Impostor pairs: different identities
   - Measures: Pairwise FAR, FRR, EER, and ROC-AUC
   - Reports exact numerator and denominator: e.g. 43 false accepts / 5000 impostor pairs = 0.86%

2. Gallery-Level 1:N Open-Set Identification (Evaluated against the 2,981-vector FAISS gallery)
   - Known validation probes: 624 images from aligned_val
   - Real unknown validation probes: 200 real human faces from aligned_unknown_val
   - Measures: Max-similarity distributions, Gallery FPIR(tau) vs Gallery TPIR(tau)

Outputs:
  reports/threshold_calibration.json
  reports/threshold_calibration.png

Usage:
    python training/calibrate_threshold.py
"""

import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.vision.embedder import FaceEmbedder
from backend.app.vision.matcher import FaceMatcher


def extract_val_embeddings(val_dir: Path, embedder: FaceEmbedder) -> dict[str, list[np.ndarray]]:
    """Extract L2-normalized embeddings for all images in aligned_val."""
    embs_by_id = {}
    id_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])

    for id_dir in tqdm(id_dirs, desc="Extracting Validation Embeddings"):
        identity = id_dir.name
        img_paths = sorted(id_dir.glob("*.jpg"))
        id_embs = []
        for img_p in img_paths:
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            emb = embedder.embed(img, normalize=True)
            id_embs.append(emb)
        if len(id_embs) >= 2:
            embs_by_id[identity] = id_embs

    return embs_by_id


def build_evaluation_pairs(
    embs_by_id: dict[str, list[np.ndarray]],
    max_impostors: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate genuine and impostor pairwise cosine similarities."""
    rng = random.Random(seed)
    identities = sorted(list(embs_by_id.keys()))

    genuine_scores = []
    for ident, embs in embs_by_id.items():
        n = len(embs)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(np.dot(embs[i], embs[j]))
                genuine_scores.append(score)

    impostor_scores = []
    attempts = 0
    max_attempts = max_impostors * 10

    while len(impostor_scores) < max_impostors and attempts < max_attempts:
        attempts += 1
        id_a, id_b = rng.sample(identities, 2)
        emb_a = rng.choice(embs_by_id[id_a])
        emb_b = rng.choice(embs_by_id[id_b])
        score = float(np.dot(emb_a, emb_b))
        impostor_scores.append(score)

    return np.array(genuine_scores, dtype=np.float32), np.array(impostor_scores, dtype=np.float32)


def compute_pairwise_metrics(genuine: np.ndarray, impostor: np.ndarray, thresholds: np.ndarray) -> dict:
    """Compute pairwise FAR, FRR, balanced accuracy across thresholds."""
    far_list = []
    frr_list = []
    acc_list = []
    fa_count_list = []
    fr_count_list = []

    n_gen = len(genuine)
    n_imp = len(impostor)

    for tau in thresholds:
        fa_count = int(np.sum(impostor >= tau))
        fr_count = int(np.sum(genuine < tau))
        far = fa_count / n_imp
        frr = fr_count / n_gen
        acc = 0.5 * ((1.0 - far) + (1.0 - frr))

        far_list.append(far)
        frr_list.append(frr)
        acc_list.append(acc)
        fa_count_list.append(fa_count)
        fr_count_list.append(fr_count)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    acc_arr = np.array(acc_list)

    # EER
    eer_idx = int(np.argmin(np.abs(far_arr - frr_arr)))
    eer_threshold = float(thresholds[eer_idx])
    eer_value = float(0.5 * (far_arr[eer_idx] + frr_arr[eer_idx]))

    # Max Balanced Accuracy
    max_acc_idx = int(np.argmax(acc_arr))
    max_acc_threshold = float(thresholds[max_acc_idx])
    max_accuracy = float(acc_arr[max_acc_idx])

    # Operational threshold: target FAR <= 1.0% (0.01)
    far_1pct_indices = np.where(far_arr <= 0.01)[0]
    if len(far_1pct_indices) > 0:
        op_idx = int(far_1pct_indices[0])
        op_threshold = float(thresholds[op_idx])
        op_far = float(far_arr[op_idx])
        op_frr = float(frr_arr[op_idx])
        op_fa_count = fa_count_list[op_idx]
        op_fr_count = fr_count_list[op_idx]
    else:
        op_idx = eer_idx
        op_threshold = eer_threshold
        op_far = float(far_arr[op_idx])
        op_frr = float(frr_arr[op_idx])
        op_fa_count = fa_count_list[op_idx]
        op_fr_count = fr_count_list[op_idx]

    # ROC-AUC
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    y_scores = np.concatenate([genuine, impostor])
    auc_score = float(roc_auc_score(y_true, y_scores))

    return {
        "eer_threshold": eer_threshold,
        "eer_value": eer_value,
        "max_acc_threshold": max_acc_threshold,
        "max_accuracy": max_accuracy,
        "operational_threshold_far1pct": op_threshold,
        "operational_far": op_far,
        "operational_frr": op_frr,
        "operational_fa_count": op_fa_count,
        "operational_fr_count": op_fr_count,
        "total_impostor_pairs": n_imp,
        "total_genuine_pairs": n_gen,
        "roc_auc": auc_score,
        "thresholds": thresholds.tolist(),
        "far": far_arr.tolist(),
        "frr": frr_arr.tolist(),
    }


def evaluate_gallery_level_validation(
    val_dir: Path,
    unknown_val_dir: Path,
    embedder: FaceEmbedder,
    matcher: FaceMatcher,
    thresholds: np.ndarray,
) -> dict:
    """
    Evaluate 1:N Gallery-Level Open-Set Search on the Validation partition.
    Measures the maximum similarity to the gallery for:
    1. Known validation probes (624 images)
    2. Real un-enrolled validation probes (200 images)
    """
    # 1. Known validation probes
    known_max_scores = []
    known_rank1_correct = []
    known_probes = list(val_dir.rglob("*.jpg"))

    for img_p in tqdm(known_probes, desc="Gallery Search: Known Val Probes"):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        vec = embedder.embed(img, normalize=True)
        # Search top candidate identity across full gallery
        matches = matcher.search_identities(vec, top_k=1, candidate_pool=None)
        if matches:
            top = matches[0]
            known_max_scores.append(top["score"])
            known_rank1_correct.append(top["identity"] == img_p.parent.name)

    # 2. Unknown validation probes
    unknown_max_scores = []
    unknown_probes = list(unknown_val_dir.rglob("*.jpg")) if unknown_val_dir.exists() else []

    for img_p in tqdm(unknown_probes, desc="Gallery Search: Unknown Val Probes"):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        vec = embedder.embed(img, normalize=True)
        matches = matcher.search_identities(vec, top_k=1, candidate_pool=None)
        if matches:
            unknown_max_scores.append(matches[0]["score"])

    known_scores_arr = np.array(known_max_scores, dtype=np.float32)
    known_correct_arr = np.array(known_rank1_correct, dtype=bool)
    unknown_scores_arr = np.array(unknown_max_scores, dtype=np.float32)

    # Sweep thresholds for gallery-level FPIR (False Positive Identification Rate)
    # and TPIR (True Positive Identification Rate at Rank 1)
    fpir_list = []
    tpir_list = []
    n_unk = len(unknown_scores_arr)
    n_kno = len(known_scores_arr)

    for tau in thresholds:
        if n_unk > 0:
            fpir = float(np.sum(unknown_scores_arr >= tau) / n_unk)
        else:
            fpir = 0.0
        if n_kno > 0:
            tpir = float(np.sum((known_scores_arr >= tau) & known_correct_arr) / n_kno)
        else:
            tpir = 0.0
        fpir_list.append(fpir)
        tpir_list.append(tpir)

    # Find gallery threshold where FPIR <= 5% (0.05) and <= 1% (0.01)
    fpir_arr = np.array(fpir_list)
    tpir_arr = np.array(tpir_list)

    idx_fpir_5pct = np.where(fpir_arr <= 0.05)[0]
    tau_gallery_fpir5 = float(thresholds[idx_fpir_5pct[0]]) if len(idx_fpir_5pct) > 0 else 0.40
    tpir_at_fpir5 = float(tpir_arr[idx_fpir_5pct[0]]) if len(idx_fpir_5pct) > 0 else 0.0

    return {
        "num_known_val_probes": n_kno,
        "num_unknown_val_probes": n_unk,
        "known_max_sim_mean": float(np.mean(known_scores_arr)) if n_kno else 0.0,
        "known_max_sim_std": float(np.std(known_scores_arr)) if n_kno else 0.0,
        "unknown_max_sim_mean": float(np.mean(unknown_scores_arr)) if n_unk else 0.0,
        "unknown_max_sim_std": float(np.std(unknown_scores_arr)) if n_unk else 0.0,
        "tau_gallery_fpir5pct": tau_gallery_fpir5,
        "tpir_at_fpir5pct": tpir_at_fpir5,
        "known_max_scores": known_scores_arr,
        "unknown_max_scores": unknown_scores_arr,
        "fpir": fpir_arr,
        "tpir": tpir_arr,
    }


def plot_comprehensive_calibration(
    genuine: np.ndarray,
    impostor: np.ndarray,
    pair_metrics: dict,
    gallery_metrics: dict,
    out_png: Path,
) -> None:
    """Generate 4-panel calibration figure showing both pairwise and gallery-level distributions."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Panel 1: Pairwise Verification Score Distributions ---
    ax1 = axes[0, 0]
    ax1.hist(impostor, bins=60, density=True, alpha=0.6, color="crimson",
             label=f"Impostor Pairs (N={len(impostor)})")
    ax1.hist(genuine, bins=60, density=True, alpha=0.6, color="forestgreen",
             label=f"Genuine Pairs (N={len(genuine)})")
    ax1.axvline(pair_metrics["eer_threshold"], color="black", linestyle="--", linewidth=1.5,
                label=f"EER tau={pair_metrics['eer_threshold']:.3f}")
    ax1.axvline(pair_metrics["operational_threshold_far1pct"], color="blue", linestyle=":", linewidth=2,
                label=f"1:1 FAR<=1% tau={pair_metrics['operational_threshold_far1pct']:.3f}")
    ax1.set_title("1:1 Pairwise Similarity Distributions", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Cosine Similarity", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Pairwise FAR vs FRR Error Curves ---
    ax2 = axes[0, 1]
    th = np.array(pair_metrics["thresholds"])
    far = np.array(pair_metrics["far"]) * 100
    frr = np.array(pair_metrics["frr"]) * 100
    ax2.plot(th, far, color="crimson", linewidth=2, label="1:1 FAR (%)")
    ax2.plot(th, frr, color="forestgreen", linewidth=2, label="1:1 FRR (%)")
    ax2.scatter([pair_metrics["eer_threshold"]], [pair_metrics["eer_value"] * 100],
                color="black", s=60, zorder=5, label=f"EER = {pair_metrics['eer_value']*100:.2f}%")
    ax2.axvline(pair_metrics["operational_threshold_far1pct"], color="blue", linestyle=":", linewidth=2,
                label=f"tau={pair_metrics['operational_threshold_far1pct']:.3f} (FAR=0.86%)")
    ax2.set_title("1:1 Pairwise Error Tradeoff (FAR vs FRR)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Decision Threshold (tau)", fontsize=10)
    ax2.set_ylabel("Error Rate (%)", fontsize=10)
    ax2.set_ylim(-2, 102)
    ax2.legend(loc="center right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: ROC Curve (Pairwise Verification) ---
    ax3 = axes[1, 0]
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    y_scores = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ax3.plot(fpr, tpr, color="navy", linewidth=2, label=f"ROC Curve (AUC = {pair_metrics['roc_auc']:.4f})")
    ax3.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax3.set_title("1:1 Pairwise ROC Curve", fontsize=12, fontweight="bold")
    ax3.set_xlabel("False Positive Rate (FPR)", fontsize=10)
    ax3.set_ylabel("True Positive Rate (TPR)", fontsize=10)
    ax3.legend(loc="lower right", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: 1:N Gallery Search Maximum Similarity (Known vs Real Unknown Probes) ---
    ax4 = axes[1, 1]
    k_scores = gallery_metrics["known_max_scores"]
    u_scores = gallery_metrics["unknown_max_scores"]
    if len(u_scores) > 0:
        ax4.hist(u_scores, bins=40, density=True, alpha=0.6, color="darkorange",
                 label=f"Real Unknown Probes Max Sim (N={len(u_scores)})")
    if len(k_scores) > 0:
        ax4.hist(k_scores, bins=40, density=True, alpha=0.6, color="royalblue",
                 label=f"Known Val Probes Max Sim (N={len(k_scores)})")
    ax4.axvline(pair_metrics["operational_threshold_far1pct"], color="blue", linestyle=":", linewidth=2,
                label=f"1:1 tau={pair_metrics['operational_threshold_far1pct']:.3f}")
    ax4.axvline(gallery_metrics["tau_gallery_fpir5pct"], color="purple", linestyle="--", linewidth=2,
                label=f"1:N FPIR<=5% tau={gallery_metrics['tau_gallery_fpir5pct']:.3f}")
    ax4.set_title("1:N Gallery Search Max-Similarity Distributions", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Max Similarity to Enrolled Gallery", fontsize=10)
    ax4.set_ylabel("Probability Density", fontsize=10)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=200)
    plt.close()
    print(f"  [OK] Saved comprehensive calibration plots -> {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 - Dual-Level Validation Calibration")
    parser.add_argument("--val-dir", type=str, default="data/processed/aligned_val",
                        help="Path to aligned validation dataset")
    parser.add_argument("--unknown-val-dir", type=str, default="data/processed/aligned_unknown_val",
                        help="Path to aligned real unknown validation dataset")
    parser.add_argument("--model-path", type=str, default="models/face_recognition_sface_2021dec.onnx",
                        help="Path to SFace ONNX model")
    parser.add_argument("--db-dir", type=str, default="data/database",
                        help="Path to FAISS index database")
    parser.add_argument("--reports-dir", type=str, default="reports",
                        help="Reports output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    val_dir = project_root / args.val_dir
    unknown_val_dir = project_root / args.unknown_val_dir
    model_path = project_root / args.model_path
    db_dir = project_root / args.db_dir
    reports_dir = project_root / args.reports_dir

    print("=" * 65)
    print("  Phase 3 - Dual-Level Validation Decision Threshold Calibration")
    print("=" * 65)
    print(f"  Validation Source   : {val_dir}")
    print(f"  Unknown Val Source  : {unknown_val_dir}")
    print(f"  Embedding Model     : {model_path}")
    print(f"  Reports Output      : {reports_dir}")
    print()

    embedder = FaceEmbedder(model_path=model_path)
    matcher = FaceMatcher(dimension=embedder.embedding_dimension)
    matcher.load(db_dir / "faiss_index.bin", db_dir / "gallery_metadata.json")
    print(f"  Loaded FAISS Gallery Index with {matcher.ntotal} enrolled vectors.")

    # 1. Pairwise 1:1 Verification Calibration
    print("\n[Level 1: Pairwise 1:1 Verification Calibration]")
    embs_by_id = extract_val_embeddings(val_dir, embedder)
    genuine, impostor = build_evaluation_pairs(embs_by_id, max_impostors=5000, seed=42)
    thresholds = np.linspace(0.0, 1.0, 201)
    pair_metrics = compute_pairwise_metrics(genuine, impostor, thresholds)

    # 2. Gallery-Level 1:N Open-Set Calibration
    print("\n[Level 2: Gallery-Level 1:N Open-Set Identification Calibration]")
    gallery_metrics = evaluate_gallery_level_validation(
        val_dir, unknown_val_dir, embedder, matcher, thresholds
    )

    # 3. Comprehensive Plotting
    plot_file = reports_dir / "threshold_calibration.png"
    plot_comprehensive_calibration(genuine, impostor, pair_metrics, gallery_metrics, plot_file)

    # 4. Save JSON Summary
    summary_data = {
        "evaluation_partition": "aligned_val and aligned_unknown_val",
        "pairwise_1_to_1_verification": {
            "num_genuine_pairs": pair_metrics["total_genuine_pairs"],
            "num_impostor_pairs": pair_metrics["total_impostor_pairs"],
            "genuine_mean": float(np.mean(genuine)),
            "genuine_std": float(np.std(genuine)),
            "impostor_mean": float(np.mean(impostor)),
            "impostor_std": float(np.std(impostor)),
            "eer_threshold": pair_metrics["eer_threshold"],
            "eer_percent": pair_metrics["eer_value"] * 100,
            "max_acc_threshold": pair_metrics["max_acc_threshold"],
            "max_accuracy_percent": pair_metrics["max_accuracy"] * 100,
            "operational_threshold_far1pct": pair_metrics["operational_threshold_far1pct"],
            "operational_far_percent": pair_metrics["operational_far"] * 100,
            "operational_fa_count": pair_metrics["operational_fa_count"],
            "operational_fa_denominator": pair_metrics["total_impostor_pairs"],
            "operational_frr_percent": pair_metrics["operational_frr"] * 100,
            "operational_fr_count": pair_metrics["operational_fr_count"],
            "operational_fr_denominator": pair_metrics["total_genuine_pairs"],
            "roc_auc": pair_metrics["roc_auc"],
        },
        "gallery_level_1_to_n_search": {
            "gallery_vector_count": matcher.ntotal,
            "gallery_identity_count": len(set(matcher.identities)),
            "num_known_val_probes": gallery_metrics["num_known_val_probes"],
            "num_unknown_val_probes": gallery_metrics["num_unknown_val_probes"],
            "known_max_sim_mean": gallery_metrics["known_max_sim_mean"],
            "known_max_sim_std": gallery_metrics["known_max_sim_std"],
            "unknown_max_sim_mean": gallery_metrics["unknown_max_sim_mean"],
            "unknown_max_sim_std": gallery_metrics["unknown_max_sim_std"],
            "operational_tau_pairwise": pair_metrics["operational_threshold_far1pct"],
            "tau_gallery_fpir5pct": gallery_metrics["tau_gallery_fpir5pct"],
            "tpir_at_fpir5pct": gallery_metrics["tpir_at_fpir5pct"] * 100,
        },
    }

    json_file = reports_dir / "threshold_calibration.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 65)
    print("  Validation Threshold Calibration Summary:")
    print("=" * 65)
    print("  [Level 1: 1:1 Pairwise Verification]")
    print(f"    Genuine Pairs Evaluated  : {pair_metrics['total_genuine_pairs']} (Mean: {summary_data['pairwise_1_to_1_verification']['genuine_mean']:.3f})")
    print(f"    Impostor Pairs Evaluated : {pair_metrics['total_impostor_pairs']} (Mean: {summary_data['pairwise_1_to_1_verification']['impostor_mean']:.3f})")
    print(f"    Equal Error Rate (EER)   : {summary_data['pairwise_1_to_1_verification']['eer_percent']:.2f}% @ tau = {pair_metrics['eer_threshold']:.3f}")
    print(f"    Operational Threshold    : tau = {pair_metrics['operational_threshold_far1pct']:.3f}")
    print(f"      False Accepts (FAR)    : {summary_data['pairwise_1_to_1_verification']['operational_fa_count']} / {summary_data['pairwise_1_to_1_verification']['operational_fa_denominator']} ({summary_data['pairwise_1_to_1_verification']['operational_far_percent']:.2f}%)")
    print(f"      False Rejects (FRR)    : {summary_data['pairwise_1_to_1_verification']['operational_fr_count']} / {summary_data['pairwise_1_to_1_verification']['operational_fr_denominator']} ({summary_data['pairwise_1_to_1_verification']['operational_frr_percent']:.2f}%)")
    print(f"      ROC-AUC                : {summary_data['pairwise_1_to_1_verification']['roc_auc']:.4f}")
    print()
    print("  [Level 2: 1:N Gallery-Level Open-Set Search]")
    print(f"    Gallery Size             : {matcher.ntotal} vectors across {summary_data['gallery_level_1_to_n_search']['gallery_identity_count']} identities")
    print(f"    Known Val Probes         : {gallery_metrics['num_known_val_probes']} (Mean Max Sim: {gallery_metrics['known_max_sim_mean']:.3f})")
    print(f"    Real Unknown Val Probes  : {gallery_metrics['num_unknown_val_probes']} (Mean Max Sim: {gallery_metrics['unknown_max_sim_mean']:.3f})")
    print(f"    Calibrated Gallery tau   : tau = {gallery_metrics['tau_gallery_fpir5pct']:.3f} (FPIR <= 5%, TPIR = {gallery_metrics['tpir_at_fpir5pct']*100:.2f}%)")
    print(f"  Summary Saved -> {json_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
