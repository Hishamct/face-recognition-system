"""
training/calibrate_threshold.py
================================
Phase 3 - Step 2: Calibrate decision threshold tau on the VALIDATION split.

Reads  : data/processed/aligned_val/<IdentityName>/*.jpg
Outputs: reports/threshold_calibration.png
         reports/threshold_calibration.json

CRITICAL EVALUATION RULE:
The decision threshold tau is calibrated exclusively on the validation set
and frozen BEFORE the held-out test set is evaluated.

Calculates:
1. Genuine vs Impostor cosine similarity distributions
2. False Acceptance Rate (FAR) and False Rejection Rate (FRR) curves
3. Equal Error Rate (EER) threshold
4. Operational security threshold targeting FAR <= 1.0%
5. Maximum Accuracy threshold
6. Receiver Operating Characteristic (ROC) curve & AUC

Usage:
    python training/calibrate_threshold.py [--val-dir data/processed/aligned_val]
"""

import argparse
import itertools
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


def extract_val_embeddings(val_dir: Path, embedder: FaceEmbedder) -> dict:
    """Extract embeddings for all images in the validation split grouped by identity."""
    id_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    embeddings_by_id = {}

    for id_dir in id_dirs:
        images = sorted(id_dir.glob("*.jpg"))
        if len(images) < 2:
            continue

        embs = []
        for img_p in images:
            img = cv2.imread(str(img_p))
            if img is not None:
                embs.append(embedder.embed(img, normalize=True))

        if len(embs) >= 2:
            embeddings_by_id[id_dir.name] = embs

    return embeddings_by_id


def build_evaluation_pairs(
    embeddings_by_id: dict,
    max_impostors: int = 5000,
    seed: int = 42,
) -> tuple:
    """
    Generate genuine (same identity) and impostor (different identity) cosine similarity scores.
    """
    rng = random.Random(seed)
    genuine_scores = []
    impostor_scores = []

    # 1. Genuine pairs: all combinations within each identity
    for id_name, embs in embeddings_by_id.items():
        for e1, e2 in itertools.combinations(embs, 2):
            sim = float(np.dot(e1, e2))
            genuine_scores.append(sim)

    # 2. Impostor pairs: sample random pairs across different identities
    identities = list(embeddings_by_id.keys())
    impostor_pairs = set()

    for _ in range(max_impostors * 2):
        id1, id2 = rng.sample(identities, 2)
        e1_idx = rng.randrange(len(embeddings_by_id[id1]))
        e2_idx = rng.randrange(len(embeddings_by_id[id2]))
        pair_key = (id1, e1_idx, id2, e2_idx) if id1 < id2 else (id2, e2_idx, id1, e1_idx)

        if pair_key not in impostor_pairs:
            impostor_pairs.add(pair_key)
            e1 = embeddings_by_id[id1][e1_idx]
            e2 = embeddings_by_id[id2][e2_idx]
            sim = float(np.dot(e1, e2))
            impostor_scores.append(sim)
            if len(impostor_scores) >= max_impostors:
                break

    return np.array(genuine_scores, dtype=np.float32), np.array(impostor_scores, dtype=np.float32)


def compute_metrics_across_thresholds(
    genuine: np.ndarray,
    impostor: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """Sweep thresholds to compute FAR, FRR, Accuracy, EER, and optimal operating points."""
    far_list = []
    frr_list = []
    acc_list = []

    n_gen = len(genuine)
    n_imp = len(impostor)

    for tau in thresholds:
        # FAR: impostors accepted as genuine
        far = np.sum(impostor >= tau) / n_imp
        # FRR: genuine rejected as impostor
        frr = np.sum(genuine < tau) / n_gen
        # Balanced accuracy
        acc = 0.5 * ((1.0 - far) + (1.0 - frr))

        far_list.append(far)
        frr_list.append(frr)
        acc_list.append(acc)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    acc_arr = np.array(acc_list)

    # 1. Equal Error Rate (EER): where |FAR - FRR| is minimal
    eer_idx = int(np.argmin(np.abs(far_arr - frr_arr)))
    eer_threshold = float(thresholds[eer_idx])
    eer_value = float(0.5 * (far_arr[eer_idx] + frr_arr[eer_idx]))

    # 2. Maximum Accuracy threshold
    max_acc_idx = int(np.argmax(acc_arr))
    max_acc_threshold = float(thresholds[max_acc_idx])
    max_accuracy = float(acc_arr[max_acc_idx])

    # 3. Operational threshold: target FAR <= 1.0% (0.01)
    far_1pct_indices = np.where(far_arr <= 0.01)[0]
    if len(far_1pct_indices) > 0:
        op_idx = int(far_1pct_indices[0])  # lowest threshold satisfying FAR <= 1%
        op_threshold = float(thresholds[op_idx])
        op_far = float(far_arr[op_idx])
        op_frr = float(frr_arr[op_idx])
    else:
        op_idx = eer_idx
        op_threshold = eer_threshold
        op_far = float(far_arr[op_idx])
        op_frr = float(frr_arr[op_idx])

    # 4. ROC-AUC
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
        "roc_auc": auc_score,
        "thresholds": thresholds.tolist(),
        "far": far_arr.tolist(),
        "frr": frr_arr.tolist(),
        "acc": acc_arr.tolist(),
    }


def plot_calibration_results(
    genuine: np.ndarray,
    impostor: np.ndarray,
    metrics: dict,
    out_png: Path,
) -> None:
    """Plot score distributions, FAR/FRR curves, and ROC curve."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Score Distributions ---
    ax1 = axes[0]
    ax1.hist(impostor, bins=60, density=True, alpha=0.6, color="crimson", label=f"Impostor (N={len(impostor)})")
    ax1.hist(genuine, bins=60, density=True, alpha=0.6, color="forestgreen", label=f"Genuine (N={len(genuine)})")
    ax1.axvline(metrics["eer_threshold"], color="black", linestyle="--", linewidth=1.5,
                label=f"EER tau={metrics['eer_threshold']:.3f}")
    ax1.axvline(metrics["operational_threshold_far1pct"], color="blue", linestyle=":", linewidth=2,
                label=f"FAR<=1% tau={metrics['operational_threshold_far1pct']:.3f}")
    ax1.set_title("Cosine Similarity Score Distributions", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Cosine Similarity", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: FAR vs FRR Error Tradeoff ---
    ax2 = axes[1]
    th = np.array(metrics["thresholds"])
    far = np.array(metrics["far"])
    frr = np.array(metrics["frr"])
    ax2.plot(th, far * 100, color="crimson", linewidth=2, label="FAR (%)")
    ax2.plot(th, frr * 100, color="forestgreen", linewidth=2, label="FRR (%)")
    ax2.scatter([metrics["eer_threshold"]], [metrics["eer_value"] * 100], color="black", s=60, zorder=5,
                label=f"EER = {metrics['eer_value']*100:.2f}%")
    ax2.axvline(metrics["operational_threshold_far1pct"], color="blue", linestyle=":", linewidth=2,
                label=f"Operational tau={metrics['operational_threshold_far1pct']:.3f}")
    ax2.set_title("Error Rates vs. Decision Threshold tau", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Decision Threshold (tau)", fontsize=10)
    ax2.set_ylabel("Error Rate (%)", fontsize=10)
    ax2.set_ylim(-2, 102)
    ax2.legend(loc="center right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: ROC Curve ---
    ax3 = axes[2]
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    y_scores = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ax3.plot(fpr, tpr, color="navy", linewidth=2, label=f"ROC Curve (AUC = {metrics['roc_auc']:.4f})")
    ax3.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax3.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("False Positive Rate (FPR)", fontsize=10)
    ax3.set_ylabel("True Positive Rate (TPR)", fontsize=10)
    ax3.legend(loc="lower right", fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=200)
    plt.close()
    print(f"  [OK] Saved calibration plots -> {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 - Calibrate Recognition Decision Threshold")
    parser.add_argument("--val-dir", type=str, default="data/processed/aligned_val",
                        help="Path to aligned validation dataset")
    parser.add_argument("--model-path", type=str, default="models/face_recognition_sface_2021dec.onnx",
                        help="Path to SFace ONNX model")
    parser.add_argument("--reports-dir", type=str, default="reports",
                        help="Directory to save calibration reports and plots")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    val_dir = project_root / args.val_dir
    model_path = project_root / args.model_path
    reports_dir = project_root / args.reports_dir

    print("=" * 65)
    print("  Phase 3 - Decision Threshold Calibration on VALIDATION Set")
    print("=" * 65)
    print(f"  Validation Source : {val_dir}")
    print(f"  Embedding Model   : {model_path}")
    print(f"  Reports Output    : {reports_dir}")
    print()

    if not val_dir.exists():
        print(f"ERROR: Validation directory not found at {val_dir}")
        raise SystemExit(1)

    embedder = FaceEmbedder(model_path=model_path)

    # 1. Extract validation embeddings
    print("Extracting validation embeddings...")
    embs_by_id = extract_val_embeddings(val_dir, embedder)
    print(f"  Extracted embeddings for {len(embs_by_id)} validation identities")

    # 2. Build Genuine and Impostor evaluation pairs
    print("Generating genuine and impostor pair similarity distributions...")
    genuine, impostor = build_evaluation_pairs(embs_by_id, max_impostors=5000, seed=42)
    print(f"  Genuine pairs evaluated  : {len(genuine)}")
    print(f"  Impostor pairs evaluated : {len(impostor)}")

    # 3. Sweep thresholds
    thresholds = np.linspace(0.0, 1.0, 201)
    metrics = compute_metrics_across_thresholds(genuine, impostor, thresholds)

    # 4. Plot and Save
    plot_file = reports_dir / "threshold_calibration.png"
    json_file = reports_dir / "threshold_calibration.json"
    plot_calibration_results(genuine, impostor, metrics, plot_file)

    # Save summary json (without full array dumps for readability)
    summary_data = {
        "dataset_split": "aligned_val",
        "num_identities_evaluated": len(embs_by_id),
        "num_genuine_pairs": len(genuine),
        "num_impostor_pairs": len(impostor),
        "genuine_mean": float(np.mean(genuine)),
        "genuine_std": float(np.std(genuine)),
        "impostor_mean": float(np.mean(impostor)),
        "impostor_std": float(np.std(impostor)),
        "eer_threshold": metrics["eer_threshold"],
        "eer_value_percent": metrics["eer_value"] * 100,
        "max_acc_threshold": metrics["max_acc_threshold"],
        "max_accuracy_percent": metrics["max_accuracy"] * 100,
        "operational_threshold_far1pct": metrics["operational_threshold_far1pct"],
        "operational_far_percent": metrics["operational_far"] * 100,
        "operational_frr_percent": metrics["operational_frr"] * 100,
        "roc_auc": metrics["roc_auc"],
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 65)
    print("  Validation Threshold Calibration Summary:")
    print("=" * 65)
    print(f"  Genuine Mean Cosine Score   : {summary_data['genuine_mean']:.3f} +/- {summary_data['genuine_std']:.3f}")
    print(f"  Impostor Mean Cosine Score  : {summary_data['impostor_mean']:.3f} +/- {summary_data['impostor_std']:.3f}")
    print(f"  Equal Error Rate (EER)      : {summary_data['eer_value_percent']:.2f}% at tau = {summary_data['eer_threshold']:.3f}")
    print(f"  Max Balanced Accuracy       : {summary_data['max_accuracy_percent']:.2f}% at tau = {summary_data['max_acc_threshold']:.3f}")
    print(f"  Operational (FAR <= 1%) tau : {summary_data['operational_threshold_far1pct']:.3f} (FAR={summary_data['operational_far_percent']:.2f}%, FRR={summary_data['operational_frr_percent']:.2f}%)")
    print(f"  ROC-AUC Score               : {summary_data['roc_auc']:.4f}")
    print(f"  Saved Calibration Summary   : {json_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
