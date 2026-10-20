"""
backend/app/vision
==================
Computer vision modules for face detection, alignment, embedding extraction, and FAISS vector matching.
"""

from backend.app.vision.aligner import FaceAligner, ARCFACE_REFERENCE_112
from backend.app.vision.detector import FaceDetection, YuNetDetector
from backend.app.vision.embedder import FaceEmbedder
from backend.app.vision.matcher import FaceMatcher

__all__ = [
    "FaceDetection",
    "YuNetDetector",
    "FaceAligner",
    "ARCFACE_REFERENCE_112",
    "FaceEmbedder",
    "FaceMatcher",
]
