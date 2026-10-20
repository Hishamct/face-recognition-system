from typing import List
from pydantic import BaseModel, Field
from app.schemas.common import BoundingBox, Landmarks5


class FaceDetectionItem(BaseModel):
    bbox: BoundingBox
    landmarks: Landmarks5
    confidence: float = Field(..., ge=0.0, le=1.0)


class DetectionResponse(BaseModel):
    success: bool
    face_count: int
    faces: List[FaceDetectionItem]
    image_width: int
    image_height: int
    latency_ms: float
