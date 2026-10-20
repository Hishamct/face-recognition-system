import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

try:
    from app.core.config import ROOT_DIR
    from app.db.models import EpochMetric, TrainingRun
    from app.schemas.training import (
        EpochMetricItem,
        TrainingHistoryResponse,
        TrainingRunItem,
        TrainingRunsResponse,
    )
except ImportError:
    from backend.app.core.config import ROOT_DIR
    from backend.app.db.models import EpochMetric, TrainingRun
    from backend.app.schemas.training import (
        EpochMetricItem,
        TrainingHistoryResponse,
        TrainingRunItem,
        TrainingRunsResponse,
    )


class TrainingService:
    """Service providing access to training runs and epoch-by-epoch loss/accuracy metrics."""

    def get_runs(self, db: Session) -> TrainingRunsResponse:
        """Lists all recorded training runs."""
        runs = db.query(TrainingRun).order_by(TrainingRun.id.desc()).all()
        items: List[TrainingRunItem] = []
        for r in runs:
            items.append(
                TrainingRunItem(
                    id=r.id,
                    run_name=r.run_name,
                    backbone=r.backbone,
                    epochs=r.epochs,
                    learning_rate=r.learning_rate,
                    batch_size=r.batch_size,
                    dropout=getattr(r, "dropout", None),
                    weight_decay=getattr(r, "weight_decay", None),
                    early_stopped_epoch=getattr(r, "early_stopped_epoch", None),
                    best_val_loss=r.best_val_loss,
                    best_val_acc=r.best_val_acc,
                    status=r.status,
                    created_at=getattr(r, "started_at", getattr(r, "created_at", datetime.now(timezone.utc))),
                )
            )
        return TrainingRunsResponse(total=len(items), runs=items)

    def get_run_history(
        self,
        db: Session,
        run_id: Optional[int] = None,
    ) -> TrainingHistoryResponse:
        """Retrieves epoch metrics for a specific run or the latest completed run."""
        query = db.query(TrainingRun)
        if run_id is not None:
            run = query.filter(TrainingRun.id == run_id).first()
        else:
            run = query.filter(TrainingRun.status == "COMPLETED").order_by(TrainingRun.id.desc()).first()
            if not run:
                run = query.order_by(TrainingRun.id.desc()).first()

        if run:
            metrics = (
                db.query(EpochMetric)
                .filter(EpochMetric.run_id == run.id)
                .order_by(EpochMetric.epoch.asc())
                .all()
            )
            epoch_items = [
                EpochMetricItem(
                    epoch=m.epoch,
                    train_loss=round(float(m.train_loss), 4),
                    train_acc=round(float(m.train_acc), 4),
                    val_loss=round(float(m.val_loss), 4),
                    val_acc=round(float(m.val_acc), 4),
                    learning_rate=round(float(getattr(m, "learning_rate", 0.002)), 6),
                )
                for m in metrics
            ]
            run_item = TrainingRunItem(
                id=run.id,
                run_name=run.run_name,
                backbone=run.backbone,
                epochs=run.epochs,
                learning_rate=run.learning_rate,
                batch_size=run.batch_size,
                dropout=getattr(run, "dropout", None),
                weight_decay=getattr(run, "weight_decay", None),
                early_stopped_epoch=getattr(run, "early_stopped_epoch", None),
                best_val_loss=run.best_val_loss,
                best_val_acc=run.best_val_acc,
                status=run.status,
                created_at=getattr(run, "started_at", getattr(run, "created_at", datetime.now(timezone.utc))),
            )
            return TrainingHistoryResponse(run=run_item, epochs=epoch_items)

        # Fallback to reports/training_history.json
        history_path = ROOT_DIR / "reports/training_history.json"
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            epochs_raw = data.get("history", [])
            epoch_items = [
                EpochMetricItem(
                    epoch=e.get("epoch", idx + 1),
                    train_loss=round(float(e.get("train_loss", 0.0)), 4),
                    train_acc=round(float(e.get("train_acc", 0.0)), 4),
                    val_loss=round(float(e.get("val_loss", 0.0)), 4),
                    val_acc=round(float(e.get("val_acc", 0.0)), 4),
                    learning_rate=round(float(e.get("learning_rate", 0.002)), 6),
                )
                for idx, e in enumerate(epochs_raw)
            ]
            run_item = TrainingRunItem(
                id=0,
                run_name=data.get("run_name", "Early-Stopped Run"),
                backbone="SFace (Frozen) + IdentityClassifier",
                epochs=data.get("epochs", len(epoch_items)),
                learning_rate=0.002,
                batch_size=32,
                dropout=0.3,
                weight_decay=0.001,
                early_stopped_epoch=data.get("early_stopped_epoch", 10),
                best_val_loss=data.get("best_val_loss", 0.2679),
                best_val_acc=data.get("best_val_acc", 0.9647),
                status="COMPLETED",
                created_at=datetime.now(timezone.utc),
            )
            return TrainingHistoryResponse(run=run_item, epochs=epoch_items)

        # Fallback dummy run if neither DB nor file exists
        run_item = TrainingRunItem(
            id=0,
            run_name="No Recorded Run",
            backbone="SFace",
            epochs=0,
            learning_rate=0.0,
            batch_size=0,
            status="EMPTY",
            created_at=datetime.now(timezone.utc),
        )
        return TrainingHistoryResponse(run=run_item, epochs=[])


training_service = TrainingService()
