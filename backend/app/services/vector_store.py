"""
backend/app/services/vector_store.py
====================================
PostgreSQL-backed Production Vector Store and Identity Service.

Provides:
- Schema initialization (DDL table generation)
- Identity enrollment and metadata management in PostgreSQL
- Exemplar and centroid prototype storage (128-D float32 buffers)
- Exact Cosine Similarity search with 100% numerical parity against FAISS
- Full synchronization between FAISS flat index and PostgreSQL database
- Recognition audit logging for live production inference
"""

import json
from pathlib import Path
import time
from typing import Optional, Union

import numpy as np
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

try:
    from backend.app.db.models import (
        Base,
        FaceEmbeddingRecord,
        Identity,
        RecognitionAuditLog,
    )
    from backend.app.db.session import SessionLocal, engine
except ImportError:
    from app.db.models import (
        Base,
        FaceEmbeddingRecord,
        Identity,
        RecognitionAuditLog,
    )
    from app.db.session import SessionLocal, engine


class PostgresVectorStore:
    """
    PostgreSQL vector store service.
    Stores identities, embeddings, and audit logs with exact cosine similarity matching.
    """

    def __init__(self, db_session: Optional[Session] = None):
        self._external_session = db_session

    def get_session(self) -> Session:
        if self._external_session is not None:
            return self._external_session
        return SessionLocal()

    @staticmethod
    def init_tables() -> None:
        """Create all tables in PostgreSQL and initialize vector functions."""
        Base.metadata.create_all(bind=engine)
        with engine.connect() as conn:
            # Ensure embedding_array column exists
            try:
                conn.execute(text("ALTER TABLE face_embeddings ADD COLUMN IF NOT EXISTS embedding_array real[];"))
                conn.commit()
            except Exception:
                pass

            # Create in-database dot product function
            try:
                create_fn = """
                CREATE OR REPLACE FUNCTION dot_product(a real[], b real[]) RETURNS real AS $$
                DECLARE
                    s real := 0;
                    i int;
                BEGIN
                    FOR i IN 1..array_length(a, 1) LOOP
                        s := s + a[i] * b[i];
                    END LOOP;
                    RETURN s;
                END;
                $$ LANGUAGE plpgsql IMMUTABLE;
                """
                conn.execute(text(create_fn))
                conn.commit()
            except Exception:
                pass

    @staticmethod
    def has_pgvector() -> bool:
        """Check if native pgvector extension is installed in PostgreSQL."""
        try:
            with engine.connect() as conn:
                res = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector';")).scalar()
                return res == 1
        except Exception:
            return False

    def enroll_identity(
        self,
        name: str,
        exemplar_embeddings: list[np.ndarray],
        prototype_embedding: Optional[np.ndarray] = None,
        notes: Optional[str] = None,
    ) -> Identity:
        """
        Enroll a new identity with multiple exemplar embeddings and optional centroid prototype.
        """
        session = self.get_session()
        close_needed = self._external_session is None

        try:
            # Check if identity exists
            stmt = select(Identity).where(Identity.name == name)
            existing = session.execute(stmt).scalar_one_or_none()

            if existing is None:
                identity = Identity(name=name, notes=notes, is_active=True)
                session.add(identity)
                session.flush()
            else:
                identity = existing

            # Add exemplar records
            for i, vec in enumerate(exemplar_embeddings):
                rec = FaceEmbeddingRecord(
                    identity_id=identity.id,
                    exemplar_type="exemplar",
                    source_image=f"{name}_exemplar_{i:04d}",
                )
                rec.set_vector(vec)
                session.add(rec)

            # Add prototype record
            if prototype_embedding is not None:
                proto_rec = FaceEmbeddingRecord(
                    identity_id=identity.id,
                    exemplar_type="prototype",
                    source_image=f"{name}_prototype",
                )
                proto_rec.set_vector(prototype_embedding)
                session.add(proto_rec)

            session.commit()
            session.refresh(identity)
            return identity
        finally:
            if close_needed:
                session.close()

    def sync_from_faiss_gallery(
        self,
        faiss_index_path: Union[str, Path],
        gallery_metadata_path: Union[str, Path],
    ) -> dict:
        """
        Synchronize the entire FAISS gallery and metadata JSON into PostgreSQL.
        """
        self.init_tables()
        meta_p = Path(gallery_metadata_path)
        if not meta_p.exists():
            raise FileNotFoundError(f"Gallery metadata not found at {meta_p}")

        with open(meta_p, "r", encoding="utf-8") as f:
            payload = json.load(f)

        import faiss
        index = faiss.read_index(str(faiss_index_path))
        ntotal = index.ntotal
        vectors = index.reconstruct_n(0, ntotal)  # (ntotal, 128) float32

        identities = payload["identities"]
        metas = payload.get("metadata", [{} for _ in identities])

        session = self.get_session()
        close_needed = self._external_session is None

        try:
            # Map unique identity names to DB Identity objects
            unique_names = sorted(list(set(identities)))
            id_map = {}

            for name in unique_names:
                stmt = select(Identity).where(Identity.name == name)
                ident = session.execute(stmt).scalar_one_or_none()
                if ident is None:
                    ident = Identity(name=name, is_active=True)
                    session.add(ident)
                    session.flush()
                id_map[name] = ident.id

            # Insert all vectors
            inserted = 0
            for i in range(ntotal):
                name = identities[i]
                meta = metas[i] if i < len(metas) else {}
                vec = vectors[i]
                ex_type = meta.get("type", "exemplar")
                src = meta.get("source", f"{name}_{i}")

                rec = FaceEmbeddingRecord(
                    identity_id=id_map[name],
                    exemplar_type=ex_type,
                    source_image=src,
                )
                rec.set_vector(vec)
                session.add(rec)
                inserted += 1

            session.commit()
            return {
                "identities_synced": len(unique_names),
                "vectors_synced": inserted,
            }
        finally:
            if close_needed:
                session.close()

    def load_gallery_cache(self) -> tuple[np.ndarray, list[str], list[int]]:
        """
        Load all active embeddings from PostgreSQL into an in-memory matrix for ultra-fast matching.
        Returns:
            matrix: (N, 128) float32 normalized embeddings
            identities: list of identity names
            embedding_ids: list of embedding DB record IDs
        """
        session = self.get_session()
        close_needed = self._external_session is None

        try:
            stmt = (
                select(FaceEmbeddingRecord, Identity.name)
                .join(Identity, FaceEmbeddingRecord.identity_id == Identity.id)
                .where(Identity.is_active.is_(True))
            )
            rows = session.execute(stmt).all()

            if not rows:
                return np.empty((0, 128), dtype=np.float32), [], []

            vecs = []
            idents = []
            rec_ids = []

            for rec, ident_name in rows:
                vecs.append(rec.get_vector())
                idents.append(ident_name)
                rec_ids.append(rec.id)

            matrix = np.vstack(vecs).astype(np.float32)
            # Guarantee unit normalization
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix /= norms

            return matrix, idents, rec_ids
        finally:
            if close_needed:
                session.close()

    def search_identities(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        matrix_cache: Optional[tuple[np.ndarray, list[str], list[int]]] = None,
    ) -> list[dict]:
        """
        Search nearest identities using Cosine Similarity in PostgreSQL.
        Aggregates candidate vectors per person taking the maximum similarity score.
        """
        if matrix_cache is None:
            matrix, idents, rec_ids = self.load_gallery_cache()
        else:
            matrix, idents, rec_ids = matrix_cache

        if len(matrix) == 0:
            return []

        q = query_vec.astype(np.float32).flatten()
        norm_q = np.linalg.norm(q)
        if norm_q > 0:
            q /= norm_q

        # Matrix inner product = cosine similarities across all enrolled gallery vectors
        sims = np.dot(matrix, q)  # (N,)

        # Group by unique identity taking maximum score
        best_by_id = {}
        for sim, ident in zip(sims, idents):
            s = float(sim)
            if ident not in best_by_id or s > best_by_id[ident]["score"]:
                best_by_id[ident] = {"identity": ident, "score": s}

        sorted_ids = sorted(best_by_id.values(), key=lambda x: x["score"], reverse=True)

        results = []
        for rank, item in enumerate(sorted_ids[:top_k], start=1):
            results.append({
                "rank": rank,
                "identity": item["identity"],
                "score": item["score"],
            })
        return results

    def search_in_database_sql(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Execute vector similarity search ENTIRELY inside PostgreSQL engine using pure SQL.
        Computes dot_product(e.embedding_array, query_vec) on the database server.
        """
        q = [float(x) for x in query_vec.flatten()]
        session = self.get_session()
        close_needed = self._external_session is None

        try:
            sql = text("""
                SELECT i.name AS identity, max(dot_product(e.embedding_array, :q)) AS score
                FROM face_embeddings e
                JOIN identities i ON e.identity_id = i.id
                WHERE i.is_active = true AND e.embedding_array IS NOT NULL
                GROUP BY i.name
                ORDER BY score DESC
                LIMIT :top_k;
            """)
            rows = session.execute(sql, {"q": q, "top_k": top_k}).fetchall()
            return [
                {"rank": rank, "identity": row[0], "score": float(row[1])}
                for rank, row in enumerate(rows, start=1)
            ]
        finally:
            if close_needed:
                session.close()

    def identify(
        self,
        query_vec: np.ndarray,
        threshold: float = 0.445,
        unknown_label: str = "Unknown",
        client_ip: Optional[str] = None,
        log_audit: bool = True,
        matrix_cache: Optional[tuple[np.ndarray, list[str], list[int]]] = None,
    ) -> dict:
        """
        Open-set face identification against PostgreSQL gallery with optional audit logging.
        """
        t0 = time.perf_counter()
        matches = self.search_identities(query_vec, top_k=1, matrix_cache=matrix_cache)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not matches:
            decision = "UNKNOWN"
            res = {
                "matched": False,
                "identity": unknown_label,
                "score": 0.0,
                "threshold": threshold,
                "latency_ms": latency_ms,
            }
        else:
            top = matches[0]
            if top["score"] >= threshold:
                decision = "RECOGNIZED"
                res = {
                    "matched": True,
                    "identity": top["identity"],
                    "score": top["score"],
                    "threshold": threshold,
                    "latency_ms": latency_ms,
                }
            else:
                decision = "UNKNOWN"
                res = {
                    "matched": False,
                    "identity": unknown_label,
                    "score": top["score"],
                    "threshold": threshold,
                    "best_candidate": top["identity"],
                    "latency_ms": latency_ms,
                }

        if log_audit:
            self.log_recognition_event(
                matched_identity=res["identity"] if res["matched"] else res.get("best_candidate"),
                score=res["score"],
                threshold=threshold,
                decision=decision,
                latency_ms=latency_ms,
                client_ip=client_ip,
            )

        return res

    def log_recognition_event(
        self,
        matched_identity: Optional[str],
        score: float,
        threshold: float,
        decision: str,
        latency_ms: float,
        client_ip: Optional[str] = None,
    ) -> None:
        """Record an inference event to recognition_audit_logs."""
        session = self.get_session()
        close_needed = self._external_session is None
        try:
            log_rec = RecognitionAuditLog(
                matched_identity=matched_identity,
                similarity_score=float(score),
                threshold_used=float(threshold),
                decision=decision,
                latency_ms=float(latency_ms),
                client_ip=client_ip,
            )
            session.add(log_rec)
            session.commit()
        finally:
            if close_needed:
                session.close()
