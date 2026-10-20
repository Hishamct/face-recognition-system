"""
backend/app/vision/embedder.py
==============================
Pretrained deep face feature embedder wrapper.
Extracts 128-D L2-normalized biometric embeddings from 112x112 aligned face crops.
"""

from pathlib import Path
from typing import Union

import cv2
import numpy as np


class FaceEmbedder:
    """
    Extracts discriminative facial feature embeddings using OpenCV's SFace model.
    Input:  Aligned 112x112 face crop (RGB or BGR numpy array)
    Output: 128-dimensional L2-normalized float32 embedding vector
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/face_recognition_sface_2021dec.onnx",
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Face recognition model not found at: {self.model_path}\n"
                "Please ensure models/face_recognition_sface_2021dec.onnx is downloaded."
            )

        self.device = device.lower()
        backend_id = cv2.dnn.DNN_BACKEND_OPENCV
        target_id = cv2.dnn.DNN_TARGET_CPU

        if self.device == "cuda" and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            backend_id = cv2.dnn.DNN_BACKEND_CUDA
            target_id = cv2.dnn.DNN_TARGET_CUDA

        self._recognizer = cv2.FaceRecognizerSF.create(
            model=str(self.model_path),
            config="",
            backend_id=backend_id,
            target_id=target_id,
        )
        self.embedding_dimension = 128

    def embed(
        self,
        aligned_face: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate feature embedding for an aligned 112x112 face crop.
        
        Args:
            aligned_face: (112, 112, 3) image in BGR or RGB format
            normalize: If True, returns unit L2-normalized vector (norm == 1.0)
            
        Returns:
            np.ndarray: 128-dimensional float32 vector with shape (128,)
        """
        if aligned_face is None or aligned_face.size == 0:
            raise ValueError("Input aligned face image is empty or None")

        if aligned_face.shape[:2] != (112, 112):
            aligned_face = cv2.resize(aligned_face, (112, 112), interpolation=cv2.INTER_LINEAR)

        # OpenCV FaceRecognizerSF expects BGR
        raw_feat = self._recognizer.feature(aligned_face)  # shape: (1, 128)
        feat = raw_feat.flatten().astype(np.float32)

        if normalize:
            norm = np.linalg.norm(feat)
            if norm > 1e-8:
                feat = feat / norm

        return feat

    def embed_batch(
        self,
        aligned_faces: list[np.ndarray],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of aligned 112x112 face crops.
        
        Returns:
            np.ndarray: (N, 128) float32 matrix
        """
        if not aligned_faces:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        embeddings = [self.embed(face, normalize=normalize) for face in aligned_faces]
        return np.vstack(embeddings).astype(np.float32)

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute Cosine Similarity between two embedding vectors.
        For L2-normalized vectors: cosine_similarity == dot_product.
        Range: [-1.0, 1.0], where 1.0 indicates identical facial identity.
        """
        v1 = embedding1.flatten()
        v2 = embedding2.flatten()
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))
