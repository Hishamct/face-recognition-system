from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

try:
    from app.db.session import get_db
    from app.schemas.evaluation import EvaluationHistoryResponse, EvaluationMetricsResponse
    from app.services.evaluation_service import evaluation_service
except ImportError:
    from backend.app.db.session import get_db
    from backend.app.schemas.evaluation import EvaluationHistoryResponse, EvaluationMetricsResponse
    from backend.app.services.evaluation_service import evaluation_service

router = APIRouter(prefix="/evaluation", tags=["Evaluation & Benchmarks"])


@router.get(
    "/metrics",
    response_model=EvaluationMetricsResponse,
    summary="Get latest evaluation benchmark metrics",
    description="Returns the latest test set metrics: Rank-1, Top-5, Top-10, MRR, TPIR, FPIR, ROC-AUC, and calibrated threshold.",
)
def get_latest_evaluation_metrics(
    db: Session = Depends(get_db),
) -> EvaluationMetricsResponse:
    return evaluation_service.get_latest_metrics(db=db)


@router.get(
    "/history",
    response_model=EvaluationHistoryResponse,
    summary="Get evaluation benchmark history",
    description="Returns all recorded evaluation benchmark runs.",
)
def get_evaluation_history(
    db: Session = Depends(get_db),
) -> EvaluationHistoryResponse:
    return evaluation_service.get_evaluation_history(db=db)
