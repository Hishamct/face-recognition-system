"""
training/train_classifier.py
============================
Phase 4: Supervised Classifier Head Training & Transfer Learning with Early Stopping.
Trains a regularized linear/MLP classification head over frozen SFace 128-D face embeddings.
Logs epoch metrics to PostgreSQL (table `epoch_metrics`), implements early stopping
based strictly on validation loss, and saves publication-ready training curves.
"""

import argparse
import copy
import datetime
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db.models import EpochMetric, TrainingRun
from backend.app.db.session import SessionLocal
from backend.app.vision.classifier import IdentityClassifier
from backend.app.vision.embedder import FaceEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainClassifier")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_augmentation_pipeline() -> A.Compose:
    """Conservative Albumentations pipeline matching Phase 1/2 standard."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.4),
        A.Affine(scale=(0.97, 1.03), translate_percent=(-0.03, 0.03), rotate=(-5, 5), border_mode=cv2.BORDER_REFLECT, p=0.3),
    ])


def load_dataset_features(
    data_dir: Path,
    embedder: FaceEmbedder,
    class_to_idx: Dict[str, int],
    augment: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from partitioned directory and extract 128-D SFace embeddings.
    If augment=True, generates augmented crops and extracts embeddings.
    """
    embeddings_list: List[np.ndarray] = []
    labels_list: List[int] = []

    aug_pipeline = build_augmentation_pipeline() if augment else None

    logger.info("Extracting embeddings from: %s (augment=%s)...", data_dir, augment)
    identities = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and d.name in class_to_idx])

    total_images = 0
    for identity in identities:
        label = class_to_idx[identity]
        id_dir = data_dir / identity
        img_paths = sorted(list(id_dir.glob("*.jpg")) + list(id_dir.glob("*.png")))

        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue

            total_images += 1
            feat = embedder.embed(img, normalize=True)
            embeddings_list.append(feat)
            labels_list.append(label)

            if augment and aug_pipeline is not None:
                augmented = aug_pipeline(image=img)["image"]
                feat_aug = embedder.embed(augmented, normalize=True)
                embeddings_list.append(feat_aug)
                labels_list.append(label)

    logger.info("Processed %d source images -> %d feature vectors.", total_images, len(embeddings_list))
    return np.array(embeddings_list, dtype=np.float32), np.array(labels_list, dtype=np.int64)


def plot_curves(history: List[dict], best_epoch: int, output_path: Path) -> None:
    """Plot publication-quality training & validation loss and accuracy curves with best checkpoint marked."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    best_record = next(h for h in history if h["epoch"] == best_epoch)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Subplot 1: Cross-Entropy Loss
    ax1.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    ax1.plot(epochs, val_loss, label="Validation Loss", color="#d62728", linewidth=2.2, marker="s", markersize=4)
    ax1.axvline(best_epoch, linestyle="--", color="gray", alpha=0.8, label=f"Best Checkpoint (Epoch {best_epoch})")
    ax1.scatter([best_epoch], [best_record["val_loss"]], color="crimson", s=100, zorder=5)
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # Subplot 2: Classification Accuracy
    ax2.plot(epochs, train_acc, label="Training Accuracy (%)", color="#2ca02c", linewidth=2.2, marker="o", markersize=4)
    ax2.plot(epochs, val_acc, label="Validation Accuracy (%)", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4)
    ax2.axvline(best_epoch, linestyle="--", color="gray", alpha=0.8, label=f"Best Checkpoint (Epoch {best_epoch})")
    ax2.scatter(
        [best_epoch],
        [best_record["val_acc"]],
        color="crimson",
        s=120,
        zorder=5,
        label=f"Best Val Acc: {best_record['val_acc']:.2f}%",
    )
    ax2.set_title("Top-1 Accuracy vs. Epochs", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim([min(min(train_acc), min(val_acc)) - 5, 102])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle("Supervised Classification Head: Transfer Learning Curves (SFace Frozen Backbone)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close()
    logger.info("Saved training curves plot to %s", output_path)


def train_classifier(
    train_dir: Path,
    val_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.002,
    weight_decay: float = 1e-3,
    dropout_p: float = 0.3,
    patience: int = 5,
    output_model: Path = Path("models/classifier_head.pt"),
    history_json: Path = Path("reports/training_history.json"),
    curve_plot: Path = Path("reports/training_curves.png"),
    save_db: bool = True,
) -> dict:
    set_seed(42)

    # 1. Discover identities strictly from training directory
    train_identities = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(train_identities)}
    num_classes = len(class_to_idx)
    logger.info("Identified %d unique identity classes.", num_classes)

    mapping_path = Path("models/class_to_idx.json")
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)
    logger.info("Saved class mapping to %s", mapping_path)

    # 2. Extract features
    embedder = FaceEmbedder(model_path="models/face_recognition_sface_2021dec.onnx", device="cpu")
    X_train_np, y_train_np = load_dataset_features(train_dir, embedder, class_to_idx, augment=True)
    X_val_np, y_val_np = load_dataset_features(val_dir, embedder, class_to_idx, augment=False)

    train_dataset = TensorDataset(torch.from_numpy(X_train_np), torch.from_numpy(y_train_np))
    val_dataset = TensorDataset(torch.from_numpy(X_val_np), torch.from_numpy(y_val_np))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model setup with regularized hyperparameters
    model = IdentityClassifier(
        embedding_dim=embedder.embedding_dimension,
        num_classes=num_classes,
        dropout_p=dropout_p,
        class_to_idx=class_to_idx,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 4. PostgreSQL Training Run Initialization
    db_session = None
    training_run = None
    if save_db:
        try:
            db_session = SessionLocal()
            run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            training_run = TrainingRun(
                run_name=f"sface_regularized_head_{run_timestamp}",
                backbone="sface_2021dec",
                num_classes=num_classes,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                status="RUNNING",
            )
            db_session.add(training_run)
            db_session.commit()
            db_session.refresh(training_run)
            logger.info("Created PostgreSQL TrainingRun ID: %d (%s)", training_run.id, training_run.run_name)
        except Exception as e:
            logger.warning("Could not initialize database logging (%s). Continuing in local mode.", e)
            if db_session:
                db_session.rollback()
                db_session.close()
            db_session = None
            training_run = None

    # 5. Training loop with early stopping tracking
    history: List[dict] = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 1
    best_state_dict = None
    patience_counter = 0
    stopped_early = False

    logger.info("=" * 76)
    logger.info(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^10} | {'Val Acc':^9} | {'LR':^10} | Status")
    logger.info("=" * 76)

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        total_train_samples = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(y_batch)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == y_batch).sum().item()
            total_train_samples += len(y_batch)

        epoch_train_loss = train_loss_total / total_train_samples
        epoch_train_acc = (train_correct / total_train_samples) * 100.0

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for X_val_b, y_val_b in val_loader:
                val_outputs = model(X_val_b)
                v_loss = criterion(val_outputs, y_val_b)
                val_loss_total += v_loss.item() * len(y_val_b)
                v_preds = torch.argmax(val_outputs, dim=1)
                val_correct += (v_preds == y_val_b).sum().item()
                total_val_samples += len(y_val_b)

        epoch_val_loss = val_loss_total / total_val_samples
        epoch_val_acc = (val_correct / total_val_samples) * 100.0
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        # Checkpoint selection strictly based on validation loss
        status_msg = ""
        if epoch_val_loss < best_val_loss - 1e-4:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            status_msg = f"* Best (Val Loss: {best_val_loss:.4f})"
        else:
            patience_counter += 1
            status_msg = f"patience {patience_counter}/{patience}"

        logger.info(
            f"{epoch:^7d} | {epoch_train_loss:^12.4f} | {epoch_train_acc:^10.2f}% | "
            f"{epoch_val_loss:^10.4f} | {epoch_val_acc:^8.2f}% | {current_lr:^10.6f} | {status_msg}"
        )

        metric_record = {
            "epoch": epoch,
            "train_loss": float(epoch_train_loss),
            "train_acc": float(epoch_train_acc),
            "val_loss": float(epoch_val_loss),
            "val_acc": float(epoch_val_acc),
            "lr": float(current_lr),
            "is_best_checkpoint": (epoch == best_epoch),
        }
        history.append(metric_record)

        if db_session and training_run:
            try:
                db_metric = EpochMetric(
                    run_id=training_run.id,
                    epoch=epoch,
                    train_loss=epoch_train_loss,
                    train_acc=epoch_train_acc,
                    val_loss=epoch_val_loss,
                    val_acc=epoch_val_acc,
                )
                db_session.add(db_metric)
                db_session.commit()
            except Exception as e:
                logger.warning("DB commit error on epoch %d: %s", epoch, e)
                db_session.rollback()

        if patience_counter >= patience:
            logger.info("Early stopping triggered at epoch %d (patience=%d exceeded).", epoch, patience)
            stopped_early = True
            break

    logger.info("=" * 76)
    logger.info("Training finished. Restoring best model from Epoch %d (Val Loss: %.4f, Val Acc: %.2f%%).",
                best_epoch, best_val_loss, best_val_acc)

    # 6. Save BEST model checkpoint (NOT the overfitted final epoch)
    assert best_state_dict is not None
    model.load_state_dict(best_state_dict)
    model.save(output_model)
    logger.info("Saved best validated checkpoint to %s", output_model)

    # 7. Finalize DB Run record
    if db_session and training_run:
        try:
            training_run.best_val_acc = float(best_val_acc)
            training_run.best_val_loss = float(best_val_loss)
            training_run.status = "COMPLETED"
            training_run.completed_at = datetime.datetime.now(datetime.timezone.utc)
            db_session.commit()
            logger.info("Updated PostgreSQL TrainingRun ID %d status to COMPLETED.", training_run.id)
            db_session.close()
        except Exception as e:
            logger.warning("Failed to finalize DB run record: %s", e)

    # 8. Save history JSON
    history_json.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "run_name": training_run.run_name if training_run else "sface_regularized_head",
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val_loss), 4),
        "best_val_acc": round(float(best_val_acc), 2),
        "regularization": {
            "dropout": dropout_p,
            "weight_decay": weight_decay,
            "learning_rate": learning_rate,
            "patience": patience,
        },
        "history": history,
    }
    with open(history_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved training history to %s", history_json)

    # 9. Plot curves
    plot_curves(history, best_epoch, curve_plot)
    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Train Supervised Classifier Head with Early Stopping")
    parser.add_argument("--train-dir", type=Path, default=Path("data/processed/aligned_train"))
    parser.add_argument("--val-dir", type=Path, default=Path("data/processed/aligned_val"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-model", type=Path, default=Path("models/classifier_head.pt"))
    parser.add_argument("--history-json", type=Path, default=Path("reports/training_history.json"))
    parser.add_argument("--curve-plot", type=Path, default=Path("reports/training_curves.png"))
    parser.add_argument("--no-db", action="store_true", help="Disable PostgreSQL database logging")
    args = parser.parse_args()

    train_classifier(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_p=args.dropout,
        patience=args.patience,
        output_model=args.output_model,
        history_json=args.history_json,
        curve_plot=args.curve_plot,
        save_db=not args.no_db,
    )


if __name__ == "__main__":
    main()
