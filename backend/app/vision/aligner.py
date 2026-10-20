"""
backend/app/vision/aligner.py
=============================
5-point facial landmark similarity transformation and normalization.
Maps faces into canonical 112x112 coordinates using standard ArcFace reference points.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


# Standard InsightFace / ArcFace canonical 5-point landmark reference for 112x112 crops
# Points are ordered: [right_eye, left_eye, nose_tip, right_mouth_corner, left_mouth_corner]
# (Note: "right eye" is from the subject's perspective, located on the left side of the image x ~ 38.3)
ARCFACE_REFERENCE_112 = np.array([
    [38.2946, 51.6963],  # Right eye
    [73.5318, 51.5014],  # Left eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Right mouth corner
    [70.7299, 92.2041],  # Left mouth corner
], dtype=np.float32)


class FaceAligner:
    """
    Performs 5-point affine similarity alignment on face images.
    Aligns eyes horizontally, centers the nose and mouth, and normalizes scale.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (112, 112),
        reference_landmarks: Optional[np.ndarray] = None,
    ):
        self.target_size = target_size
        if reference_landmarks is None:
            self.reference_landmarks = ARCFACE_REFERENCE_112.copy()
            if target_size != (112, 112):
                # Scale reference points proportionally if custom target size is requested
                scale_x = target_size[0] / 112.0
                scale_y = target_size[1] / 112.0
                self.reference_landmarks[:, 0] *= scale_x
                self.reference_landmarks[:, 1] *= scale_y
        else:
            self.reference_landmarks = reference_landmarks.astype(np.float32)

    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        fallback_bbox: Optional[list[int]] = None,
    ) -> np.ndarray:
        """
        Align a face image to canonical coordinates using 5 facial landmarks.
        
        Args:
            image: Source image as np.ndarray (H, W, 3)
            landmarks: (5, 2) array of detected landmark points
            fallback_bbox: Optional [x, y, w, h] to use if affine estimation fails
            
        Returns:
            Aligned face crop of shape (target_size[1], target_size[0], 3)
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        pts = landmarks.astype(np.float32)

        # Estimate 2D similarity transform (rotation + uniform scale + translation)
        # cv2.estimateAffinePartial2D guarantees no non-rigid shearing
        M, _ = cv2.estimateAffinePartial2D(
            pts,
            self.reference_landmarks,
            method=cv2.LMEDS,
        )

        if M is not None:
            aligned = cv2.warpAffine(
                image,
                M,
                self.target_size,
                borderMode=cv2.BORDER_REFLECT101,
            )
            return aligned

        # Graceful fallback: crop bounding box and resize
        if fallback_bbox is not None:
            x, y, w, h = fallback_bbox
            h_img, w_img = image.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                return cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Extreme fallback: resize entire image
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
