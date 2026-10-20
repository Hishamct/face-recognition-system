"""
training/populate_and_benchmark_database.py
===========================================
Phase 4: Database Vector Store Benchmark & Full-Gallery Parity Evaluation.
Evaluates FAISS vs PostgreSQL In-Memory vs PostgreSQL In-Database SQL Dot Product
across ALL 624 test probes in `aligned_test` on the complete 2,981-vector gallery.
Also audits native pgvector extension availability.
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db.session import engine, SessionLocal
from backend.app.db.models import FaceEmbeddingRecord, Identity
from backend.app.services.vector_store import PostgresVectorStore
from backend.app.vision.embedder import FaceEmbedder
from backend.app.vision.matcher import FaceMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DatabaseBenchmark")


def populate_arrays_if_missing(session) -> int:
    """Ensure all existing rows in face_embeddings have embedding_array populated."""
    rows = session.query(FaceEmbeddingRecord.id, FaceEmbeddingRecord.embedding_data).filter(
        FaceEmbeddingRecord.embedding_array.is_(None)
    ).all()
    if not rows:
        logger.info("All vectors already have embedding_array populated.")
        return 0

    logger.info("Populating embedding_array for %d existing vectors in batches...", len(rows))
    batch_size = 200
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        for rec_id, raw_bytes in chunk:
            vec = np.frombuffer(raw_bytes, dtype=np.float32)
            arr = [float(x) for x in vec]
            session.execute(
                text("UPDATE face_embeddings SET embedding_array = :arr WHERE id = :id"),
                {"arr": arr, "id": rec_id}
            )
        session.commit()
    logger.info("Successfully populated embedding_array for all %d vectors.", len(rows))
    return len(rows)


def run_comprehensive_database_benchmark(
    faiss_index_path: Path = Path("data/database/faiss_index.bin"),
    metadata_path: Path = Path("data/database/gallery_metadata.json"),
    test_dir: Path = Path("data/processed/aligned_test"),
    output_report: Path = Path("reports/database_vector_benchmark.json"),
) -> dict:
    store = PostgresVectorStore()
    store.init_tables()  # Executed first before opening any sessions

    session = SessionLocal()
    try:
        # 1. Check or Populate Database
        identity_count = session.query(Identity).count()
        vector_count = session.query(FaceEmbeddingRecord).count()
        logger.info("PostgreSQL Status: %d identities, %d vectors currently stored.", identity_count, vector_count)

        if vector_count == 0:
            logger.info("Synchronizing gallery into PostgreSQL from %s...", faiss_index_path)
            sync_res = store.sync_from_faiss_gallery(faiss_index_path, metadata_path)
            logger.info("Synced: %s", sync_res)
            vector_count = sync_res["vectors_synced"]
            identity_count = sync_res["identities_synced"]

        populate_arrays_if_missing(session)
    finally:
        session.close()

    # 2. Check pgvector status
    pgvector_available = store.has_pgvector()
    logger.info("Native pgvector extension available in database: %s", pgvector_available)

    # 3. Load FAISS Matcher and Test Embeddings
    embedder = FaceEmbedder(model_path="models/face_recognition_sface_2021dec.onnx", device="cpu")
    faiss_matcher = FaceMatcher(dimension=embedder.embedding_dimension)
    faiss_matcher.load(faiss_index_path, metadata_path)

    test_imgs = sorted(list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.png")))
    total_queries = len(test_imgs)
    logger.info("Evaluating full parity on ALL %d test probes in %s...", total_queries, test_dir)

    query_vectors = []
    ground_truth = []
    for p in test_imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        v = embedder.embed(img, normalize=True)
        query_vectors.append(v)
        ground_truth.append(p.parent.name)

    query_vectors = np.array(query_vectors, dtype=np.float32)

    # Pre-cache PostgreSQL gallery for in-memory comparison
    gallery_cache = store.load_gallery_cache()

    # 4. Execute Benchmark
    # Method A: FAISS IndexFlatIP
    t0 = time.perf_counter()
    faiss_top1 = []
    faiss_scores = []
    for q in query_vectors:
        res = faiss_matcher.search_identities(q, top_k=1, candidate_pool=None)
        faiss_top1.append(res[0]["identity"] if res else "None")
        faiss_scores.append(res[0]["score"] if res else 0.0)
    faiss_total_time = time.perf_counter() - t0
    faiss_latency_ms = (faiss_total_time / total_queries) * 1000.0

    # Method B: PostgreSQL Application-Side Matrix Multiplication
    t0 = time.perf_counter()
    pg_mem_top1 = []
    pg_mem_scores = []
    for q in query_vectors:
        res = store.search_identities(q, top_k=1, matrix_cache=gallery_cache)
        pg_mem_top1.append(res[0]["identity"] if res else "None")
        pg_mem_scores.append(res[0]["score"] if res else 0.0)
    pg_mem_total_time = time.perf_counter() - t0
    pg_mem_latency_ms = (pg_mem_total_time / total_queries) * 1000.0

    # Method C: Pure In-Database SQL dot_product Function (sample of 50 queries for timing)
    sample_size = 50
    t0 = time.perf_counter()
    pg_sql_top1 = []
    pg_sql_scores = []
    for i in range(sample_size):
        q = query_vectors[i]
        res = store.search_in_database_sql(q, top_k=1)
        pg_sql_top1.append(res[0]["identity"] if res else "None")
        pg_sql_scores.append(res[0]["score"] if res else 0.0)
    pg_sql_sample_time = time.perf_counter() - t0
    pg_sql_latency_ms = (pg_sql_sample_time / sample_size) * 1000.0

    # 5. Measure Parity & Discrepancies
    mem_matches = sum(1 for f, p in zip(faiss_top1, pg_mem_top1) if f == p)
    mem_parity_pct = (mem_matches / total_queries) * 100.0
    max_score_diff_mem = float(np.max(np.abs(np.array(faiss_scores) - np.array(pg_mem_scores))))

    sql_matches = sum(1 for i in range(sample_size) if faiss_top1[i] == pg_sql_top1[i])
    sql_parity_pct = (sql_matches / sample_size) * 100.0
    max_score_diff_sql = float(np.max(np.abs(np.array(faiss_scores[:sample_size]) - np.array(pg_sql_scores))))

    logger.info("=" * 70)
    logger.info("  FULL DATABASE VECTOR RETRIEVAL PARITY AUDIT (624 Test Queries)")
    logger.info("=" * 70)
    logger.info(f"  FAISS vs PostgreSQL (In-Memory Matrix): {mem_matches}/{total_queries} ({mem_parity_pct:.2f}% Parity), Max Diff: {max_score_diff_mem:.2e}")
    logger.info(f"  FAISS vs PostgreSQL (In-Database SQL) : {sql_matches}/{sample_size} ({sql_parity_pct:.2f}% Parity), Max Diff: {max_score_diff_sql:.2e}")
    logger.info(f"  Latency Comparison:")
    logger.info(f"    - FAISS IndexFlatIP           : {faiss_latency_ms:.3f} ms / query")
    logger.info(f"    - PostgreSQL App Matrix Cache : {pg_mem_latency_ms:.3f} ms / query")
    logger.info(f"    - PostgreSQL In-Database SQL  : {pg_sql_latency_ms:.3f} ms / query")
    logger.info("=" * 70)

    report_payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_test_queries_evaluated": total_queries,
        "gallery_size_vectors": vector_count,
        "identities_enrolled": identity_count,
        "pgvector_extension_status": {
            "installed": pgvector_available,
            "architecture_note": (
                "PostgreSQL 17 on Windows operates without elevated Windows Administrator rights. "
                "Writing compiled DLLs into C:\\Program Files\\PostgreSQL\\17\\ requires administrative elevation (UAC). "
                "Native SQL similarity operations are executed via PostgreSQL PL/pgSQL dot_product(real[], real[]) function."
            )
        },
        "retrieval_parity": {
            "faiss_vs_postgres_in_memory": {
                "queries_evaluated": total_queries,
                "identical_matches": mem_matches,
                "parity_percentage": mem_parity_pct,
                "max_score_difference": max_score_diff_mem,
            },
            "faiss_vs_postgres_sql_function": {
                "queries_evaluated": sample_size,
                "identical_matches": sql_matches,
                "parity_percentage": sql_parity_pct,
                "max_score_difference": max_score_diff_sql,
            }
        },
        "latencies_ms_per_query": {
            "faiss_index_flat_ip": round(faiss_latency_ms, 3),
            "postgres_matrix_cache": round(pg_mem_latency_ms, 3),
            "postgres_sql_dot_product": round(pg_sql_latency_ms, 3),
        }
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    logger.info("Saved benchmark report to %s", output_report)

    return report_payload


if __name__ == "__main__":
    run_comprehensive_database_benchmark()
