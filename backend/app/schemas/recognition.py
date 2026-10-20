from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.common import BoundingBox, Landmarks5


class CandidateItem(BaseModel):
    identity: str
    confidence: float
    rank: int


class FaceRecognitionItem(BaseModel):
    bbox: BoundingBox
    landmarks: Landmarks5
    identity: str
    confidence: float
    status: str = Field(..., description="'RECOGNIZED' or 'UNKNOWN'")
    is_known: bool = False
    threshold_used: float
    top_candidates: List[CandidateItem] = Field(default_factory=list)


class RecognitionResponse(BaseModel):
    success: bool
    face_count: int
    recognized_count: int
    unknown_count: int
    faces: List[FaceRecognitionItem]
    image_width: int
    image_height: int
    latency_ms: float


class RecognitionAuditItem(BaseModel):
    id: int
    matched_identity: Optional[str]
    similarity_score: float
    threshold_used: float
    decision: str
    query_latency_ms: float
    created_at: datetime
