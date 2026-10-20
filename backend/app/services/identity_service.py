from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

try:
    from app.db.models import FaceEmbeddingRecord, Identity, RecognitionAuditLog
    from app.schemas.identities import (
        IdentityDeleteResponse,
        IdentityEnrollResponse,
        IdentityListResponse,
        IdentityResponse,
    )
    from app.services.recognition_service import recognition_service
    from app.services.vector_store import PostgresVectorStore
except ImportError:
    from backend.app.db.models import FaceEmbeddingRecord, Identity, RecognitionAuditLog
    from backend.app.schemas.identities import (
        IdentityDeleteResponse,
        IdentityEnrollResponse,
        IdentityListResponse,
        IdentityResponse,
    )
    from backend.app.services.recognition_service import recognition_service
    from backend.app.services.vector_store import PostgresVectorStore


class IdentityService:
    """Service handling Identity lifecycle, enrollment with face embeddings, listing, and deletion."""

    def __init__(self):
        self._vector_store: Optional[PostgresVectorStore] = None

    @property
    def vector_store(self) -> PostgresVectorStore:
        if self._vector_store is None:
            self._vector_store = PostgresVectorStore()
        return self._vector_store

    def list_identities(
        self,
        db: Session,
        page: int = 1,
        page_size: int = 20,
        search: Optional[str] = None,
    ) -> IdentityListResponse:
        """Lists enrolled identities with pagination and embedding counts."""
        query = db.query(Identity)
        if search:
            query = query.filter(Identity.name.ilike(f"%{search}%"))

        total = query.count()
        offset = max(0, (page - 1) * page_size)
        identities = query.order_by(Identity.name.asc()).offset(offset).limit(page_size).all()

        items = []
        for ident in identities:
            emb_count = (
                db.query(func.count(FaceEmbeddingRecord.id))
                .filter(FaceEmbeddingRecord.identity_id == ident.id)
                .scalar()
                or 0
            )
            items.append(
                IdentityResponse(
                    id=ident.id,
                    name=ident.name,
                    is_active=ident.is_active,
                    notes=ident.notes,
                    created_at=ident.created_at,
                    embedding_count=emb_count,
                )
            )

        return IdentityListResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=items,
        )

    def get_identity(self, db: Session, identity_id: int) -> Optional[IdentityResponse]:
        """Gets detailed identity record by ID."""
        ident = db.query(Identity).filter(Identity.id == identity_id).first()
        if not ident:
            return None

        emb_count = (
            db.query(func.count(FaceEmbeddingRecord.id))
            .filter(FaceEmbeddingRecord.identity_id == ident.id)
            .scalar()
            or 0
        )
        return IdentityResponse(
            id=ident.id,
            name=ident.name,
            is_active=ident.is_active,
            notes=ident.notes,
            created_at=ident.created_at,
            embedding_count=emb_count,
        )

    def enroll_identity(
        self,
        db: Session,
        name: str,
        image_bytes: bytes,
        notes: Optional[str] = None,
    ) -> IdentityEnrollResponse:
        """
        Enrolls a new face exemplar for an identity:
        Detects face -> Aligns -> Extracts SFace 128-D vector -> Inserts to DB -> Refreshes Gallery.
        """
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Identity name cannot be blank.")

        # 1. Decode image
        image = recognition_service.decode_image(image_bytes)

        # 2. Detect face
        detections = recognition_service.detector.detect(image)
        if not detections:
            raise ValueError(
                "No face detected in the enrollment image. Please provide a clear, frontal face photo."
            )

        # Pick detection with highest score
        primary_face = max(detections, key=lambda d: d.score)

        # 3. Canonical 5-point alignment
        aligned_face = recognition_service.aligner.align(image, primary_face.landmarks)

        # 4. Extract 128-D L2-normalized embedding
        embedding = recognition_service.embedder.extract(aligned_face)

        # 5. Enroll in PostgreSQL via PostgresVectorStore
        assigned_identity = self.vector_store.enroll_identity(
            name=clean_name,
            exemplar_embeddings=[embedding],
            prototype_embedding=embedding,
            notes=notes,
        )
        assigned_id = assigned_identity.id

        # Total embeddings for this identity
        emb_count = (
            db.query(func.count(FaceEmbeddingRecord.id))
            .filter(FaceEmbeddingRecord.identity_id == assigned_id)
            .scalar()
            or 1
        )

        return IdentityEnrollResponse(
            success=True,
            identity_id=assigned_id,
            name=clean_name,
            message=f"Successfully enrolled face embedding for '{clean_name}'.",
            embedding_count=emb_count,
        )

    def delete_identity(self, db: Session, identity_id: int) -> IdentityDeleteResponse:
        """Deletes an identity, all associated face embeddings, and related audit records."""
        ident = db.query(Identity).filter(Identity.id == identity_id).first()
        if not ident:
            raise ValueError(f"Identity with ID {identity_id} not found.")

        name = ident.name

        # Delete embeddings
        db.query(FaceEmbeddingRecord).filter(FaceEmbeddingRecord.identity_id == identity_id).delete()
        # Delete related audit logs
        db.query(RecognitionAuditLog).filter(RecognitionAuditLog.matched_identity == name).delete()
        # Delete identity
        db.delete(ident)
        db.commit()

        # Invalidate in-memory vector cache
        self.vector_store._gallery_matrix = None
        self.vector_store._gallery_metadata = []

        return IdentityDeleteResponse(
            success=True,
            identity_id=identity_id,
            message=f"Successfully deleted identity '{name}' and all associated vector records.",
        )


identity_service = IdentityService()
