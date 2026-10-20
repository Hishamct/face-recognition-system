"""
backend/app/vision/matcher.py
=============================
FAISS vector search matcher for face recognition.
Uses IndexFlatIP for exact cosine similarity matching on L2-normalized embeddings.
"""

import json
from pathlib import Path
from typing import Optional, Union

import faiss
import numpy as np


class FaceMatcher:
    """
    FAISS-based vector index for face identification and gallery search.
    Enrolls facial embeddings and searches nearest identities via Cosine Similarity.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = int(dimension)
        # IndexFlatIP computes exact inner-product (equivalent to cosine similarity for unit vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.identities: list[str] = []
        self.metadata: list[dict] = []

    @property
    def ntotal(self) -> int:
        """Total number of enrolled embedding vectors."""
        return self.index.ntotal

    def add(
        self,
        identity: str,
        embedding: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Add a single L2-normalized face embedding to the gallery.
        
        Args:
            identity: Person name / identity label
            embedding: (128,) or (1, 128) float32 array
            metadata: Optional extra fields (source image, timestamp, etc.)
            
        Returns:
            int: The index ID of the enrolled vector
        """
        vec = embedding.reshape(1, self.dimension).astype(np.float32).copy()
        faiss.normalize_L2(vec)

        idx_id = self.ntotal
        self.index.add(vec)
        self.identities.append(identity)
        self.metadata.append(metadata or {})
        return idx_id

    def add_batch(
        self,
        identities: list[str],
        embeddings: np.ndarray,
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """
        Add a batch of face embeddings to the gallery.
        """
        if len(identities) != len(embeddings):
            raise ValueError(f"Identities ({len(identities)}) and embeddings ({len(embeddings)}) length mismatch")

        vecs = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(vecs)

        self.index.add(vecs)
        self.identities.extend(identities)
        if metadatas is not None:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in range(len(identities))])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search the gallery for the top-k nearest neighbors.
        
        Args:
            query_embedding: (128,) or (1, 128) float32 query vector
            top_k: Number of nearest neighbors to retrieve
            
        Returns:
            List of dicts: [{"rank": 1, "identity": str, "score": float, "metadata": dict}, ...]
        """
        if self.ntotal == 0:
            return []

        k = min(top_k, self.ntotal)
        vec = query_embedding.reshape(1, self.dimension).astype(np.float32).copy()
        faiss.normalize_L2(vec)

        # Distances D: inner product (cosine similarity), Indices I: row index in gallery
        D, I = self.index.search(vec, k)

        results = []
        for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
            if idx < 0 or idx >= len(self.identities):
                continue
            results.append({
                "rank": rank,
                "identity": self.identities[idx],
                "score": float(score),
                "metadata": self.metadata[idx],
            })
        return results

    def identify(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.36,
        unknown_label: str = "Unknown",
    ) -> dict:
        """
        Perform 1:N closed/open-set face identification against gallery.
        If top-1 match score >= threshold, returns matched identity; else returns 'Unknown'.
        """
        matches = self.search(query_embedding, top_k=1)
        if not matches:
            return {
                "matched": False,
                "identity": unknown_label,
                "score": 0.0,
                "threshold": threshold,
            }

        top = matches[0]
        if top["score"] >= threshold:
            return {
                "matched": True,
                "identity": top["identity"],
                "score": top["score"],
                "threshold": threshold,
                "metadata": top["metadata"],
            }

        return {
            "matched": False,
            "identity": unknown_label,
            "score": top["score"],
            "threshold": threshold,
            "best_candidate": top["identity"],
        }

    def save(
        self,
        index_path: Union[str, Path],
        metadata_path: Union[str, Path],
    ) -> None:
        """Persist FAISS index and identity metadata to disk."""
        idx_p = Path(index_path)
        meta_p = Path(metadata_path)
        idx_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(idx_p))
        payload = {
            "dimension": self.dimension,
            "ntotal": self.ntotal,
            "identities": self.identities,
            "metadata": self.metadata,
        }
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load(
        self,
        index_path: Union[str, Path],
        metadata_path: Union[str, Path],
    ) -> None:
        """Load FAISS index and identity metadata from disk."""
        idx_p = Path(index_path)
        meta_p = Path(metadata_path)
        if not idx_p.exists():
            raise FileNotFoundError(f"FAISS index file not found at {idx_p}")
        if not meta_p.exists():
            raise FileNotFoundError(f"Gallery metadata file not found at {meta_p}")

        self.index = faiss.read_index(str(idx_p))
        with open(meta_p, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.dimension = payload["dimension"]
        self.identities = payload["identities"]
        self.metadata = payload.get("metadata", [{} for _ in self.identities])

    def clear(self) -> None:
        """Reset index and enrolled gallery records."""
        self.index.reset()
        self.identities.clear()
        self.metadata.clear()
