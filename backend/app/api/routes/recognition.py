from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

try:
    from app.db.session import get_db
    from app.schemas.recognition import RecognitionAuditItem, RecognitionResponse
    from app.services.recognition_service import recognition_service
except ImportError:
    from backend.app.db.session import get_db
    from backend.app.schemas.recognition import RecognitionAuditItem, RecognitionResponse
    from backend.app.services.recognition_service import recognition_service

router = APIRouter(prefix="/recognition", tags=["Face Recognition"])


@router.post(
    "/image",
    response_model=RecognitionResponse,
    summary="Recognize identities in an uploaded image",
    description="Detects, aligns to 112x112, extracts SFace 128-D embeddings, and classifies identities using the calibrated threshold.",
)
async def recognize_faces_in_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, BMP)"),
    threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Optional custom confidence threshold (default: calibrated 0.5697)",
    ),
    use_classifier: bool = Query(
        True,
        description="If True, uses trained PyTorch classifier head; if False, uses vector store",
    ),
    db: Session = Depends(get_db),
) -> RecognitionResponse:
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
        return recognition_service.recognize_faces(
            image_bytes=image_bytes,
            threshold_override=threshold,
            use_classifier=use_classifier,
            db=db,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Biometric recognition error: {str(exc)}",
        )


@router.get(
    "/audit-logs",
    response_model=List[RecognitionAuditItem],
    summary="Get recent recognition audit logs",
    description="Returns recent biometric recognition queries, matched identities, confidence scores, and latencies.",
)
def get_audit_logs(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of log entries to retrieve"),
    db: Session = Depends(get_db),
) -> List[RecognitionAuditItem]:
    return recognition_service.get_audit_logs(db=db, limit=limit)
