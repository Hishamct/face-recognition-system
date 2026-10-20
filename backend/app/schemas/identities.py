from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class IdentityCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    notes: Optional[str] = None


class IdentityResponse(BaseModel):
    id: int
    name: str
    is_active: bool
    notes: Optional[str] = None
    created_at: datetime
    embedding_count: int = 0


class IdentityListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[IdentityResponse]


class IdentityEnrollResponse(BaseModel):
    success: bool
    identity_id: int
    name: str
    message: str
    vector_id: Optional[int] = None
    embedding_count: int


class IdentityDeleteResponse(BaseModel):
    success: bool
    identity_id: int
    message: str
