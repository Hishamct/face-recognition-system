"""
training/evaluate_classifier.py
===============================
Phase 4: Multi-Metric Evaluation Benchmark for Supervised Face Classifier Head.
Computes Top-1/Top-5 accuracy, MRR, Macro/Weighted F1, multi-class ROC-AUC,
Precision-Recall curves, Confusion Matrix, and Open-Set Rejection benchmark.
Saves visual plots to `reports/evaluation/` and records to PostgreSQL.
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
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

# Project root path setup
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


def load_probe_embeddings(
    data_dir: Path,
    embedder: FaceEmbedder,
    class_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Extract 128-D embeddings from test set probe images.
    Returns: (embeddings, ground_truth_labels, identity_names, file_paths)
    """
    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    identities: List[str] = []
    paths: List[str] = []

    subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name in class_to_idx])
    for sdir in subdirs:
        identity = sdir.name
        label = class_to_idx[identity]
        img_files = sorted(list(sdir.glob("*.jpg")) + list(sdir.glob("*.png")))
        for f in img_files:
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

    files = sorted(list(unknown_dir.rglob("*.jpg")) + list(unknown_dir.rglob("*.png")))
    for f in files:
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
    # Row-normalize to show class recall
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

    # Compute micro-average ROC curve and ROC area
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # Compute macro-average ROC curve and ROC area
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

    plt.title("Cumulative Match Characteristic (CMC) Curve", fontsize=13, fontweight="bold", pad=12)
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


def evaluate_classifier(
    model_path: Path = Path("models/classifier_head.pt"),
    test_dir: Path = Path("data/processed/aligned_test"),
    unknown_dir: Path = Path("data/processed/aligned_unknown_test"),
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
    top1_correct = (top1_preds == y_test).sum()
    rank1_acc = (top1_correct / num_known) * 100.0

    # Compute ranks and reciprocal ranks
    ranks = []
    top5_correct = 0
    reciprocal_ranks = []
    for i in range(num_known):
        gt = y_test[i]
        sorted_indices = np.argsort(probs[i])[::-1]
        rank = int(np.where(sorted_indices == gt)[0][0]) + 1
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
        if rank <= 5:
            top5_correct += 1

    top5_acc = (top5_correct / num_known) * 100.0
    mrr = float(np.mean(reciprocal_ranks))

    logger.info("Known Test Identification: Rank-1 Acc = %.2f%% (%d/%d), Top-5 Acc = %.2f%% (%d/%d), MRR = %.4f",
                rank1_acc, top1_correct, num_known, top5_acc, top5_correct, num_known, mrr)

    # 4. Generate visual plots
    cm_plot_path = reports_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_test, top1_preds, num_classes, cm_plot_path)

    roc_plot_path = reports_dir / "roc_curve.png"
    roc_auc_micro, roc_auc_macro = plot_roc_curves(y_test, probs, num_classes, roc_plot_path)

    pr_plot_path = reports_dir / "precision_recall_curve.png"
    pr_auc_micro = plot_precision_recall_curves(y_test, probs, num_classes, pr_plot_path)

    cmc_plot_path = reports_dir / "cmc_curve.png"
    cmc_dict = plot_cmc_curve(ranks, num_known, cmc_plot_path)

    # 5. Open-Set Rejection Analysis on 200 real un-enrolled unknown face probes
    num_unknown = 0
    tpir = 0.0
    fpir = 0.0
    calibrated_threshold = 0.0

    if unknown_dir.exists():
        logger.info("Evaluating Open-Set Rejection on unknown probes from: %s", unknown_dir)
        X_unknown, unknown_paths = load_unknown_embeddings(unknown_dir, embedder)
        num_unknown = len(X_unknown)

        with torch.no_grad():
            un_tensor = torch.from_numpy(X_unknown).float()
            un_logits = model(un_tensor)
            un_probs = F.softmax(un_logits, dim=1).numpy()

        known_max_probs = np.max(probs, axis=1)
        unknown_max_probs = np.max(un_probs, axis=1)

        # Calibrate confidence threshold to achieve FPIR <= 0.05 (5%)
        # FPIR: fraction of unknown probes accepted with confidence >= threshold
        candidate_thresholds = np.linspace(0.01, 0.99, 100)
        chosen_thresh = 0.50
        for thresh in candidate_thresholds:
            fpir_val = float((unknown_max_probs >= thresh).sum() / num_unknown)
            if fpir_val <= 0.05:
                chosen_thresh = float(thresh)
                break

        calibrated_threshold = round(chosen_thresh, 4)
        fpir = float((unknown_max_probs >= calibrated_threshold).sum() / num_unknown) * 100.0

        # TPIR: fraction of known probes correctly identified with confidence >= threshold
        tpir_correct = ((top1_preds == y_test) & (known_max_probs >= calibrated_threshold)).sum()
        tpir = float(tpir_correct / num_known) * 100.0

        logger.info("Open-Set Rejection @ Threshold %.4f: FPIR = %.2f%% (%d/%d false accepts), TPIR = %.2f%% (%d/%d correct enrolls)",
                    calibrated_threshold, fpir, (unknown_max_probs >= calibrated_threshold).sum(), num_unknown,
                    tpir, tpir_correct, num_known)

    # 6. Save comprehensive evaluation summary
    summary = {
        "evaluation_name": "sface_classifier_head_test_benchmark",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_path": str(model_path),
        "test_dataset": str(test_dir),
        "num_classes": num_classes,
        "num_known_probes": num_known,
        "num_unknown_probes": num_unknown,
        "metrics": {
            "rank1_accuracy_percent": round(rank1_acc, 2),
            "top5_accuracy_percent": round(top5_acc, 2),
            "mean_reciprocal_rank_mrr": round(mrr, 4),
            "roc_auc_micro": round(roc_auc_micro, 4),
            "roc_auc_macro": round(roc_auc_macro, 4),
            "pr_auc_micro": round(pr_auc_micro, 4),
            "calibrated_confidence_threshold": calibrated_threshold,
            "open_set_tpir_percent": round(tpir, 2),
            "open_set_fpir_percent": round(fpir, 2),
            "cmc_ranks": cmc_dict,
        },
        "plot_artifacts": {
            "confusion_matrix": str(cm_plot_path),
            "roc_curve": str(roc_plot_path),
            "precision_recall_curve": str(pr_plot_path),
            "cmc_curve": str(cmc_plot_path),
        },
    }

    summary_file = reports_dir / "evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved evaluation summary JSON to %s", summary_file)

    # 7. Record benchmark in PostgreSQL
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
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/evaluation"))
    parser.add_argument("--no-db", action="store_true", help="Disable PostgreSQL database logging")
    args = parser.parse_args()

    evaluate_classifier(
        model_path=args.model_path,
        test_dir=args.test_dir,
        unknown_dir=args.unknown_dir,
        reports_dir=args.reports_dir,
        save_db=not args.no_db,
    )


if __name__ == "__main__":
    main()
