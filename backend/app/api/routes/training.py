from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

try:
    from app.db.session import get_db
    from app.schemas.training import TrainingHistoryResponse, TrainingRunsResponse
    from app.services.training_service import training_service
except ImportError:
    from backend.app.db.session import get_db
    from backend.app.schemas.training import TrainingHistoryResponse, TrainingRunsResponse
    from backend.app.services.training_service import training_service

router = APIRouter(prefix="/training", tags=["Training & Experiment Tracking"])


@router.get(
    "/runs",
    response_model=TrainingRunsResponse,
    summary="List training experiment runs",
    description="Returns all recorded training runs with hyperparameters, status, and best validation metrics.",
)
def get_training_runs(
    db: Session = Depends(get_db),
) -> TrainingRunsResponse:
    return training_service.get_runs(db=db)


@router.get(
    "/history",
    response_model=TrainingHistoryResponse,
    summary="Get epoch metrics for a training run",
    description="Returns epoch-by-epoch train/val loss and accuracy curves for the specified run or latest completed run.",
)
def get_training_history(
    run_id: Optional[int] = Query(None, description="Optional training run ID (defaults to latest completed run)"),
    db: Session = Depends(get_db),
) -> TrainingHistoryResponse:
    return training_service.get_run_history(db=db, run_id=run_id)
