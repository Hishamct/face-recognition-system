"""
training/evaluate_classifier.py
===============================
Phase 4: Multi-Metric Evaluation Benchmark for Supervised Face Classifier Head.
Reconciles Rank-1, Top-5, Rank-10, and MRR from per-query rank outputs.
Applies frozen validation-calibrated threshold to out-of-sample test probes (zero leakage).
Saves plots to `reports/evaluation/` and records benchmarks to PostgreSQL.
"""

import argparse
import datetime
import json
import logging
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db.models import EvaluationBenchmarkRecord
from backend.app.db.session import SessionLocal
from backend.app.vision.classifier import IdentityClassifier
from backend.app.vision.embedder import FaceEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EvaluateClassifier")


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score interval for binomial proportions."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    lower = max(0.0, center - spread) * 100.0
    upper = min(1.0, center + spread) * 100.0
    return round(lower, 2), round(upper, 2)


def load_probe_embeddings(
    data_dir: Path,
    embedder: FaceEmbedder,
    class_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Extract 128-D embeddings from test set probe images."""
    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    identities: List[str] = []
    paths: List[str] = []

    subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name in class_to_idx])
    for sdir in subdirs:
        identity = sdir.name
        label = class_to_idx[identity]
        for f in sorted(list(sdir.glob("*.jpg")) + list(sdir.glob("*.png"))):
            img = cv2.imread(str(f))
            if img is None:
                continue
            emb = embedder.embed(img, normalize=True)
            embeddings.append(emb)
            labels.append(label)
            identities.append(identity)
            paths.append(str(f))

    return (
        np.array(embeddings, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        identities,
        paths,
    )


def load_unknown_embeddings(
    unknown_dir: Path,
    embedder: FaceEmbedder,
) -> Tuple[np.ndarray, List[str]]:
    """Extract embeddings from un-enrolled unknown face probes."""
    embeddings: List[np.ndarray] = []
    paths: List[str] = []

    for f in sorted(list(unknown_dir.rglob("*.jpg")) + list(unknown_dir.rglob("*.png"))):
        img = cv2.imread(str(f))
        if img is None:
            continue
        emb = embedder.embed(img, normalize=True)
        embeddings.append(emb)
        paths.append(str(f))

    return np.array(embeddings, dtype=np.float32), paths


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, output_path: Path) -> None:
    """Plot high-resolution confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1e-8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8), dpi=300)
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.title(f"Normalized Confusion Matrix ({num_classes} Identity Classes)", fontsize=13, fontweight="bold", pad=12)
    plt.colorbar(label="Class Recall / Prediction Probability")
    plt.xlabel("Predicted Class Index", fontsize=11)
    plt.ylabel("True Class Index", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close()
    logger.info("Saved Confusion Matrix plot to %s", output_path)


def plot_roc_curves(
    y_true: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    output_path: Path,
) -> Tuple[float, float]:
    """Compute and plot micro-average and macro-average One-vs-Rest ROC curves."""
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:, i], probs[:, i])[0] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= num_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr_micro, tpr_micro, label=f"Micro-Average ROC (AUC = {roc_auc_micro:.4f})", color="deeppink", linestyle=":", linewidth=3)
    plt.plot(all_fpr, mean_tpr, label=f"Macro-Average ROC (AUC = {roc_auc_macro:.4f})", color="navy", linestyle="--", linewidth=2.5)
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Chance Level (AUC = 0.5000)")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (TPR)", fontsize=11)
    plt.title("Multi-Class One-vs-Rest ROC Curves", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC curve to %s (Micro AUC: %.4f, Macro AUC: %.4f)", output_path, roc_auc_micro, roc_auc_macro)

    return float(roc_auc_micro), float(roc_auc_macro)


def plot_precision_recall_curves(
    y_true: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    output_path: Path,
) -> float:
    """Compute and plot micro-average Precision-Recall curve."""
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    precision_micro, recall_micro, _ = precision_recall_curve(y_bin.ravel(), probs.ravel())
    pr_auc_micro = auc(recall_micro, precision_micro)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(recall_micro, precision_micro, color="#2ca02c", lw=2.5, label=f"Micro-Average PR (AUC = {pr_auc_micro:.4f})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Multi-Class Precision-Recall Curve", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close()
    logger.info("Saved Precision-Recall curve to %s (Micro PR-AUC: %.4f)", output_path, pr_auc_micro)

    return float(pr_auc_micro)


def plot_cmc_curve(ranks: List[int], total_queries: int, output_path: Path) -> Dict[str, float]:
    """Plot Cumulative Match Characteristic (CMC) curve for Rank-1 to Rank-10."""
    max_k = 10
    cmc_rates = []
    cmc_dict = {}

    for k in range(1, max_k + 1):
        count_top_k = sum(1 for r in ranks if r <= k)
        rate = (count_top_k / total_queries) * 100.0
        cmc_rates.append(rate)
        cmc_dict[f"Rank-{k}"] = round(rate, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5), dpi=300)
    ks = list(range(1, max_k + 1))
    plt.plot(ks, cmc_rates, marker="o", color="#1f77b4", linewidth=2.2, markersize=6)
    for k, rate in zip(ks, cmc_rates):
        plt.text(k, rate + 0.3, f"{rate:.1f}%", ha="center", fontsize=9, fontweight="bold")

    plt.title("Cumulative Match Characteristic (CMC) Curve (624 Test Queries)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Rank (k)", fontsize=11)
    plt.ylabel("Identification Rate (%)", fontsize=11)
    plt.xticks(ks)
    plt.ylim([min(cmc_rates) - 2, 102])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close()
    logger.info("Saved CMC curve to %s", output_path)

    return cmc_dict


def plot_rank_histogram(rank_counts: Dict[str, int], total_queries: int, output_path: Path) -> None:
    """Plot exact integer query count histogram across candidate ranks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(rank_counts.keys())
    counts = list(rank_counts.values())

    plt.figure(figsize=(10, 5), dpi=300)
    bars = plt.bar(labels, counts, color="#4c72b0", edgecolor="black", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            plt.text(bar.get_x() + bar.get_width() / 2.0, h + 3, str(h), ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.title("Per-Query Identification Rank Distribution (Total: 624 Test Queries)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Rank of Ground Truth Identity (from Softmax Probabilities)", fontsize=11)
    plt.ylabel("Number of Test Queries", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close()
    logger.info("Saved rank histogram plot to %s", output_path)


def evaluate_classifier(
    model_path: Path = Path("models/classifier_head.pt"),
    test_dir: Path = Path("data/processed/aligned_test"),
    unknown_dir: Path = Path("data/processed/aligned_unknown_test"),
    calibration_json: Path = Path("reports/classifier_threshold_calibration.json"),
    reports_dir: Path = Path("reports/evaluation"),
    save_db: bool = True,
) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trained classifier model
    logger.info("Loading classifier checkpoint from: %s", model_path)
    model = IdentityClassifier.load(model_path, device="cpu")
    class_to_idx = model.class_to_idx
    num_classes = model.num_classes
    logger.info("Loaded model with %d identity classes.", num_classes)

    # 2. Extract features from test set
    embedder = FaceEmbedder(model_path="models/face_recognition_sface_2021dec.onnx", device="cpu")
    logger.info("Extracting embeddings for known test probes from: %s", test_dir)
    X_test, y_test, identities, file_paths = load_probe_embeddings(test_dir, embedder, class_to_idx)
    num_known = len(y_test)
    logger.info("Extracted %d known test probe embeddings across %d classes.", num_known, num_classes)

    # 3. Model inference on known test probes
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test).float()
        logits = model(test_tensor)
        probs = F.softmax(logits, dim=1).numpy()

    top1_preds = np.argmax(probs, axis=1)
    known_confidences = np.max(probs, axis=1)
    top1_correct = int((top1_preds == y_test).sum())
    rank1_acc = (top1_correct / num_known) * 100.0

    # 4. Compute exact per-query rank integer breakdown
    ranks = []
    top5_correct = 0
    top10_correct = 0
    reciprocal_ranks = []
    rank_counts = {f"Rank-{k}": 0 for k in range(1, 11)}
    rank_counts["Rank > 10"] = 0

    for i in range(num_known):
        gt = y_test[i]
        sorted_indices = np.argsort(probs[i])[::-1]
        rank = int(np.where(sorted_indices == gt)[0][0]) + 1
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)

        if rank <= 5:
            top5_correct += 1
        if rank <= 10:
            top10_correct += 1

        if rank <= 10:
            rank_counts[f"Rank-{rank}"] += 1
        else:
            rank_counts["Rank > 10"] += 1

    top5_acc = (top5_correct / num_known) * 100.0
    top10_acc = (top10_correct / num_known) * 100.0
    mrr = float(np.mean(reciprocal_ranks))

    logger.info("=" * 70)
    logger.info("  EXACT CLOSED-SET RANKING RECONCILIATION (%d Probes)", num_known)
    logger.info("=" * 70)
    logger.info("  Rank-1 Accuracy  : %.2f%% (%d / %d)", rank1_acc, top1_correct, num_known)
    logger.info("  Top-5 Accuracy   : %.2f%% (%d / %d)", top5_acc, top5_correct, num_known)
    logger.info("  Top-10 Accuracy  : %.2f%% (%d / %d)", top10_acc, top10_correct, num_known)
    logger.info("  MRR              : %.4f", mrr)
    logger.info("  Rank Histogram   : %s", rank_counts)
    logger.info("=" * 70)

    # 5. Generate visual plots
    cm_plot_path = reports_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_test, top1_preds, num_classes, cm_plot_path)

    roc_plot_path = reports_dir / "roc_curve.png"
    roc_auc_micro, roc_auc_macro = plot_roc_curves(y_test, probs, num_classes, roc_plot_path)

    pr_plot_path = reports_dir / "precision_recall_curve.png"
    pr_auc_micro = plot_precision_recall_curves(y_test, probs, num_classes, pr_plot_path)

    cmc_plot_path = reports_dir / "cmc_curve.png"
    cmc_dict = plot_cmc_curve(ranks, num_known, cmc_plot_path)

    hist_plot_path = reports_dir / "rank_histogram.png"
    plot_rank_histogram(rank_counts, num_known, hist_plot_path)

    # 6. Out-of-Sample Open-Set Evaluation with Frozen Validation Threshold
    calibrated_threshold = 0.9000
    if calibration_json.exists():
        with open(calibration_json, "r", encoding="utf-8") as f:
            calib_meta = json.load(f)
            calibrated_threshold = float(calib_meta.get("calibrated_threshold", 0.9000))
        logger.info("Loaded FROZEN validation threshold: %.4f from %s", calibrated_threshold, calibration_json)
    else:
        logger.warning("Calibration file %s not found. Using default threshold %.4f", calibration_json, calibrated_threshold)

    num_unknown = 0
    tpir = 0.0
    fpir = 0.0
    fpir_ci = (0.0, 0.0)
    tpir_ci = (0.0, 0.0)
    false_accepts = 0
    correct_accepts = 0

    if unknown_dir.exists():
        logger.info("Evaluating Open-Set Rejection out-of-sample on: %s", unknown_dir)
        X_unknown, unknown_paths = load_unknown_embeddings(unknown_dir, embedder)
        num_unknown = len(X_unknown)

        with torch.no_grad():
            un_logits = model(torch.from_numpy(X_unknown).float())
            un_probs = F.softmax(un_logits, dim=1).numpy()

        unknown_confidences = np.max(un_probs, axis=1)

        # Apply FROZEN threshold strictly out-of-sample
        false_accepts = int((unknown_confidences >= calibrated_threshold).sum())
        fpir = (false_accepts / num_unknown) * 100.0
        fpir_ci = wilson_score_interval(false_accepts, num_unknown)

        correct_accepts = int(((top1_preds == y_test) & (known_confidences >= calibrated_threshold)).sum())
        tpir = (correct_accepts / num_known) * 100.0
        tpir_ci = wilson_score_interval(correct_accepts, num_known)

        logger.info("=" * 70)
        logger.info("  OUT-OF-SAMPLE OPEN-SET EVALUATION @ FROZEN THRESHOLD %.4f", calibrated_threshold)
        logger.info("=" * 70)
        logger.info("  FPIR on Unknowns : %.2f%% (%d / %d) [95%% Wilson CI: %.2f%% - %.2f%%]",
                    fpir, false_accepts, num_unknown, fpir_ci[0], fpir_ci[1])
        logger.info("  TPIR on Knowns   : %.2f%% (%d / %d) [95%% Wilson CI: %.2f%% - %.2f%%]",
                    tpir, correct_accepts, num_known, tpir_ci[0], tpir_ci[1])
        logger.info("=" * 70)

    # 7. Save comprehensive evaluation summary
    summary = {
        "evaluation_name": "sface_classifier_head_test_benchmark",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_path": str(model_path),
        "test_dataset": str(test_dir),
        "unknown_dataset": str(unknown_dir),
        "num_classes": num_classes,
        "num_known_probes": num_known,
        "num_unknown_probes": num_unknown,
        "metrics": {
            "rank1_accuracy_percent": round(rank1_acc, 2),
            "top5_accuracy_percent": round(top5_acc, 2),
            "top10_accuracy_percent": round(top10_acc, 2),
            "mean_reciprocal_rank_mrr": round(mrr, 4),
            "roc_auc_micro": round(roc_auc_micro, 4),
            "roc_auc_macro": round(roc_auc_macro, 4),
            "pr_auc_micro": round(pr_auc_micro, 4),
            "calibrated_confidence_threshold": calibrated_threshold,
            "threshold_selection_protocol": "Calibrated exclusively on validation partitions (zero test data leakage)",
            "open_set_fpir_percent": round(fpir, 2),
            "open_set_fpir_95_ci": list(fpir_ci),
            "open_set_fpir_false_accepts_count": false_accepts,
            "open_set_tpir_percent": round(tpir, 2),
            "open_set_tpir_95_ci": list(tpir_ci),
            "open_set_tpir_correct_accepts_count": correct_accepts,
            "rank_histogram": rank_counts,
            "cmc_ranks": cmc_dict,
        },
        "plot_artifacts": {
            "confusion_matrix": str(cm_plot_path),
            "roc_curve": str(roc_plot_path),
            "precision_recall_curve": str(pr_plot_path),
            "cmc_curve": str(cmc_plot_path),
            "rank_histogram": str(hist_plot_path),
        },
    }

    summary_file = reports_dir / "evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved evaluation summary JSON to %s", summary_file)

    # 8. Record benchmark in PostgreSQL
    if save_db:
        try:
            db_session = SessionLocal()
            record = EvaluationBenchmarkRecord(
                evaluation_name="sface_classifier_head_test_benchmark",
                test_split="aligned_test",
                rank1_acc=float(rank1_acc),
                top5_acc=float(top5_acc),
                mrr=float(mrr),
                tpir=float(tpir),
                fpir=float(fpir),
                num_known_probes=int(num_known),
                num_unknown_probes=int(num_unknown),
            )
            db_session.add(record)
            db_session.commit()
            db_session.refresh(record)
            logger.info("Saved benchmark to PostgreSQL `evaluation_benchmarks` record ID: %d", record.id)
            db_session.close()
        except Exception as e:
            logger.warning("Could not record benchmark in PostgreSQL: %s", e)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Supervised Face Classifier Head")
    parser.add_argument("--model-path", type=Path, default=Path("models/classifier_head.pt"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/processed/aligned_test"))
    parser.add_argument("--unknown-dir", type=Path, default=Path("data/processed/aligned_unknown_test"))
    parser.add_argument("--calibration-json", type=Path, default=Path("reports/classifier_threshold_calibration.json"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/evaluation"))
    parser.add_argument("--no-db", action="store_true", help="Disable PostgreSQL database logging")
    args = parser.parse_args()

    evaluate_classifier(
        model_path=args.model_path,
        test_dir=args.test_dir,
        unknown_dir=args.unknown_dir,
        calibration_json=args.calibration_json,
        reports_dir=args.reports_dir,
        save_db=not args.no_db,
    )


if __name__ == "__main__":
    main()
