from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

try:
    from app.db.session import get_db
    from app.schemas.identities import (
        IdentityDeleteResponse,
        IdentityEnrollResponse,
        IdentityListResponse,
        IdentityResponse,
    )
    from app.services.identity_service import identity_service
except ImportError:
    from backend.app.db.session import get_db
    from backend.app.schemas.identities import (
        IdentityDeleteResponse,
        IdentityEnrollResponse,
        IdentityListResponse,
        IdentityResponse,
    )
    from backend.app.services.identity_service import identity_service

router = APIRouter(prefix="/identities", tags=["Identities & Enrollment"])


@router.get(
    "",
    response_model=IdentityListResponse,
    summary="List enrolled identities",
    description="Returns paginated list of enrolled individuals with embedding counts.",
)
def list_identities(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by identity name"),
    db: Session = Depends(get_db),
) -> IdentityListResponse:
    return identity_service.list_identities(
        db=db,
        page=page,
        page_size=page_size,
        search=search,
    )


@router.get(
    "/{identity_id}",
    response_model=IdentityResponse,
    summary="Get identity details",
    description="Returns detailed information for a single enrolled identity by ID.",
)
def get_identity(
    identity_id: int,
    db: Session = Depends(get_db),
) -> IdentityResponse:
    identity = identity_service.get_identity(db=db, identity_id=identity_id)
    if not identity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Identity with ID {identity_id} not found.",
        )
    return identity


@router.post(
    "",
    response_model=IdentityEnrollResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a new identity",
    description="Uploads a face image, detects & aligns the face, extracts a 128-D SFace embedding, and saves to PostgreSQL & FAISS.",
)
async def enroll_identity(
    name: str = Form(..., description="Full name or unique identifier of the person"),
    notes: Optional[str] = Form(None, description="Optional metadata or notes"),
    file: UploadFile = File(..., description="Frontal face photo for enrollment"),
    db: Session = Depends(get_db),
) -> IdentityEnrollResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{file.content_type}'. Please upload an image file.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )

    try:
        return identity_service.enroll_identity(
            db=db,
            name=name,
            image_bytes=image_bytes,
            notes=notes,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enrollment failed: {str(exc)}",
        )


@router.delete(
    "/{identity_id}",
    response_model=IdentityDeleteResponse,
    summary="Delete an enrolled identity",
    description="Removes an identity and all associated vector embeddings from PostgreSQL and FAISS.",
)
def delete_identity(
    identity_id: int,
    db: Session = Depends(get_db),
) -> IdentityDeleteResponse:
    try:
        return identity_service.delete_identity(db=db, identity_id=identity_id)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete identity: {str(exc)}",
        )
