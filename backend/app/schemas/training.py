from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class EpochMetricItem(BaseModel):
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    learning_rate: Optional[float] = None


class TrainingRunItem(BaseModel):
    id: int
    run_name: str
    backbone: str
    epochs: int
    learning_rate: float
    batch_size: int
    dropout: Optional[float] = None
    weight_decay: Optional[float] = None
    early_stopped_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None
    best_val_acc: Optional[float] = None
    status: str
    created_at: datetime


class TrainingRunsResponse(BaseModel):
    total: int
    runs: List[TrainingRunItem]


class TrainingHistoryResponse(BaseModel):
    run: TrainingRunItem
    epochs: List[EpochMetricItem]
