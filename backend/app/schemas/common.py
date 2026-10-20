from typing import Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int = Field(..., description="Top-left X coordinate")
    y1: int = Field(..., description="Top-left Y coordinate")
    x2: int = Field(..., description="Bottom-right X coordinate")
    y2: int = Field(..., description="Bottom-right Y coordinate")
    width: int = Field(..., description="Bounding box width")
    height: int = Field(..., description="Bounding box height")
    confidence: float = Field(..., description="Face detector confidence score")


class LandmarkPoint(BaseModel):
    x: float = Field(..., description="Landmark X coordinate")
    y: float = Field(..., description="Landmark Y coordinate")


class Landmarks5(BaseModel):
    left_eye: LandmarkPoint
    right_eye: LandmarkPoint
    nose_tip: LandmarkPoint
    mouth_left: LandmarkPoint
    mouth_right: LandmarkPoint


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    database_connected: bool
