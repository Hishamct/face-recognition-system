from fastapi import APIRouter, File, HTTPException, UploadFile, status

try:
    from app.schemas.detection import DetectionResponse
    from app.services.recognition_service import recognition_service
except ImportError:
    from backend.app.schemas.detection import DetectionResponse
    from backend.app.services.recognition_service import recognition_service

router = APIRouter(prefix="/detection", tags=["Face Detection"])


@router.post(
    "/image",
    response_model=DetectionResponse,
    summary="Detect faces in an uploaded image",
    description="Runs deep learning YuNet detector to find face bounding boxes and 5-point facial landmarks.",
)
async def detect_faces_in_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, BMP)")
) -> DetectionResponse:
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
        return recognition_service.detect_faces(image_bytes)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face detection processing error: {str(exc)}",
        )
