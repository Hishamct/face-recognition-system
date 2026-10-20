"""
training/calibrate_classifier_threshold.py
==========================================
Phase 4: Documented Threshold Calibration on Validation Data (Zero Test Leakage).
Calibrates the open-set rejection threshold for the supervised classification head
strictly using `aligned_val` (known validation probes) and `aligned_unknown_val`
(un-enrolled validation probes). The calibrated threshold is frozen and saved to
`reports/classifier_threshold_calibration.json` for out-of-sample test evaluation.
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.vision.classifier import IdentityClassifier
from backend.app.vision.embedder import FaceEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CalibrateClassifierThreshold")


def load_probe_embeddings(
    data_dir: Path,
    embedder: FaceEmbedder,
    class_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract 128-D embeddings from validation directory."""
    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    names: List[str] = []

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
            names.append(identity)

    return np.array(embeddings, dtype=np.float32), np.array(labels, dtype=np.int64), names


def load_unknown_embeddings(
    unknown_dir: Path,
    embedder: FaceEmbedder,
) -> np.ndarray:
    """Extract embeddings from un-enrolled validation probes."""
    embeddings: List[np.ndarray] = []
    for f in sorted(list(unknown_dir.rglob("*.jpg")) + list(unknown_dir.rglob("*.png"))):
        img = cv2.imread(str(f))
        if img is None:
            continue
        emb = embedder.embed(img, normalize=True)
        embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)


def calibrate_threshold(
    model_path: Path = Path("models/classifier_head.pt"),
    val_dir: Path = Path("data/processed/aligned_val"),
    unknown_val_dir: Path = Path("data/processed/aligned_unknown_val"),
    output_json: Path = Path("reports/classifier_threshold_calibration.json"),
    output_plot: Path = Path("reports/classifier_threshold_calibration.png"),
    target_fpir: float = 0.05,
) -> dict:
    logger.info("Loading classifier from %s...", model_path)
    model = IdentityClassifier.load(model_path, device="cpu")
    class_to_idx = model.class_to_idx
    num_classes = model.num_classes

    embedder = FaceEmbedder(model_path="models/face_recognition_sface_2021dec.onnx", device="cpu")

    # 1. Extract validation features (STRICTLY VALIDATION DATA - ZERO TEST LEAKAGE)
    logger.info("Extracting known validation probes from: %s", val_dir)
    X_val, y_val, names = load_probe_embeddings(val_dir, embedder, class_to_idx)
    num_known_val = len(y_val)

    logger.info("Extracting unknown validation probes from: %s", unknown_val_dir)
    X_un_val = load_unknown_embeddings(unknown_val_dir, embedder)
    num_unknown_val = len(X_un_val)

    logger.info("Validation Dataset: %d known probes across %d classes, %d unknown probes.",
                num_known_val, num_classes, num_unknown_val)

    # 2. Compute classifier softmax probabilities
    with torch.no_grad():
        known_logits = model(torch.from_numpy(X_val).float())
        known_probs = F.softmax(known_logits, dim=1).numpy()

        un_logits = model(torch.from_numpy(X_un_val).float())
        un_probs = F.softmax(un_logits, dim=1).numpy()

    known_preds = np.argmax(known_probs, axis=1)
    known_confidences = np.max(known_probs, axis=1)
    unknown_confidences = np.max(un_probs, axis=1)

    # 3. Sweep candidate confidence thresholds
    candidate_thresholds = np.linspace(0.01, 0.99, 500)
    fpir_curve = []
    tpir_curve = []

    best_thresh = None
    for thresh in candidate_thresholds:
        # FPIR: fraction of unknown probes accepted because max softmax prob >= threshold
        fpir = float((unknown_confidences >= thresh).sum() / num_unknown_val)
        # TPIR: fraction of known probes correctly classified and confidence >= threshold
        tpir = float(((known_preds == y_val) & (known_confidences >= thresh)).sum() / num_known_val)

        fpir_curve.append(fpir)
        tpir_curve.append(tpir)

        if fpir <= target_fpir and best_thresh is None:
            best_thresh = float(thresh)

    if best_thresh is None:
        best_thresh = float(candidate_thresholds[np.argmin(fpir_curve)])

    calibrated_thresh = round(best_thresh, 4)
    calibrated_fpir = float((unknown_confidences >= calibrated_thresh).sum() / num_unknown_val) * 100.0
    calibrated_tpir = float(((known_preds == y_val) & (known_confidences >= calibrated_thresh)).sum() / num_known_val) * 100.0

    logger.info("=" * 65)
    logger.info("  CLASSIFIER THRESHOLD CALIBRATION RESULTS (VALIDATION DATA ONLY)")
    logger.info("=" * 65)
    logger.info(f"  Target FPIR Policy       : <= {target_fpir * 100:.1f}%")
    logger.info(f"  Calibrated Threshold     : {calibrated_thresh:.4f}")
    logger.info(f"  Validation FPIR Achieved : {calibrated_fpir:.2f}% ({(unknown_confidences >= calibrated_thresh).sum()}/{num_unknown_val} false accepts)")
    logger.info(f"  Validation TPIR Achieved : {calibrated_tpir:.2f}% ({((known_preds == y_val) & (known_confidences >= calibrated_thresh)).sum()}/{num_known_val} correct accepts)")
    logger.info("=" * 65)

    # 4. Plot Calibration Tradeoff
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(candidate_thresholds, np.array(tpir_curve) * 100.0, label="Validation TPIR (% Correct Accepts)", color="#2ca02c", lw=2.2)
    plt.plot(candidate_thresholds, np.array(fpir_curve) * 100.0, label="Validation FPIR (% False Accepts)", color="#d62728", lw=2.2)
    plt.axvline(calibrated_thresh, linestyle="--", color="black", lw=1.5, label=f"Calibrated Threshold ({calibrated_thresh:.4f})")
    plt.axhline(target_fpir * 100.0, linestyle=":", color="crimson", alpha=0.7, label=f"Target FPIR Ceiling ({target_fpir * 100:.0f}%)")

    plt.title("Classifier Open-Set Threshold Calibration on Validation Partition", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Softmax Confidence Threshold (tau)", fontsize=11)
    plt.ylabel("Rate (%)", fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([-2.0, 102.0])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="center right", frameon=True)
    plt.tight_layout()
    plt.savefig(str(output_plot), bbox_inches="tight")
    plt.close()
    logger.info("Saved threshold calibration plot to %s", output_plot)

    # 5. Save Calibration Metadata JSON
    calibration_data = {
        "score_type": "maximum_softmax_probability",
        "calibration_dataset": {
            "known_validation_probes": num_known_val,
            "known_validation_path": str(val_dir),
            "unknown_validation_probes": num_unknown_val,
            "unknown_validation_path": str(unknown_val_dir),
        },
        "optimization_objective": f"Find min tau such that validation FPIR <= {target_fpir}",
        "calibrated_threshold": calibrated_thresh,
        "validation_metrics": {
            "tpir_percent": round(calibrated_tpir, 2),
            "fpir_percent": round(calibrated_fpir, 2),
            "false_accepts_count": int((unknown_confidences >= calibrated_thresh).sum()),
            "correct_accepts_count": int(((known_preds == y_val) & (known_confidences >= calibrated_thresh)).sum()),
        },
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol_note": "Calibrated exclusively on validation partitions. The test set was NOT accessed during calibration."
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2)
    logger.info("Saved calibration metadata to %s", output_json)

    return calibration_data


def main():
    parser = argparse.ArgumentParser(description="Calibrate Classifier Rejection Threshold on Validation Data")
    parser.add_argument("--model-path", type=Path, default=Path("models/classifier_head.pt"))
    parser.add_argument("--val-dir", type=Path, default=Path("data/processed/aligned_val"))
    parser.add_argument("--unknown-val-dir", type=Path, default=Path("data/processed/aligned_unknown_val"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/classifier_threshold_calibration.json"))
    parser.add_argument("--output-plot", type=Path, default=Path("reports/classifier_threshold_calibration.png"))
    parser.add_argument("--target-fpir", type=float, default=0.05)
    args = parser.parse_args()

    calibrate_threshold(
        model_path=args.model_path,
        val_dir=args.val_dir,
        unknown_val_dir=args.unknown_val_dir,
        output_json=args.output_json,
        output_plot=args.output_plot,
        target_fpir=args.target_fpir,
    )


if __name__ == "__main__":
    main()
