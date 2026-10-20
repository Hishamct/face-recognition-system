from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class EvaluationMetricsResponse(BaseModel):
    id: Optional[int] = None
    evaluation_name: str
    model_version: str
    gallery_version: str
    threshold_used: float
    rank1_acc: float
    top5_acc: float
    top10_acc: Optional[float] = None
    mrr: float
    tpir: float
    fpir: float
    micro_auc: Optional[float] = None
    macro_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    tested_at: Optional[datetime] = None


class EvaluationRunResponse(BaseModel):
    success: bool
    message: str
    metrics: EvaluationMetricsResponse


class EvaluationHistoryResponse(BaseModel):
    total: int
    items: List[EvaluationMetricsResponse]
