"""
backend/app/vision/classifier.py
================================
Supervised PyTorch classification head for face identity recognition.
Operates on top of frozen 128-D SFace feature embeddings.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityClassifier(nn.Module):
    """
    Supervised Linear/MLP classification head.
    Maps 128-dimensional L2-normalized face embeddings to identity class probabilities.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_classes: int = 150,
        dropout_p: float = 0.2,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.class_to_idx = class_to_idx or {}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.net = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits."""
        return self.net(x)

    def predict_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[str, float, List[Dict[str, Union[str, float, int]]]]:
        """
        Predict identity given a single 128-D embedding vector.

        Args:
            embedding: (128,) float32 vector
            top_k: number of top ranked candidates to return

        Returns:
            Tuple of (top1_identity_name, top1_confidence, list_of_top_k_candidates)
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(embedding).float()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            logits = self.forward(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        top1_idx = int(top_indices[0])
        top1_name = self.idx_to_class.get(top1_idx, f"Class_{top1_idx}")
        top1_prob = float(probs[top1_idx])

        top_candidates = [
            {
                "identity": self.idx_to_class.get(int(idx), f"Class_{idx}"),
                "probability": float(probs[idx]),
                "rank": rank + 1,
            }
            for rank, idx in enumerate(top_indices)
        ]
        return top1_name, top1_prob, top_candidates

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model weights and metadata to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
            "class_to_idx": self.class_to_idx,
        }
        torch.save(payload, str(filepath))

    @classmethod
    def load(cls, filepath: Union[str, Path], device: str = "cpu") -> "IdentityClassifier":
        """Load trained model weights and class mapping from checkpoint file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found at: {filepath}")

        checkpoint = torch.load(str(filepath), map_location=device, weights_only=False)
        model = cls(
            embedding_dim=checkpoint["embedding_dim"],
            num_classes=checkpoint["num_classes"],
            class_to_idx=checkpoint.get("class_to_idx", {}),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model

    # Method alias
    load_checkpoint = load
