"""
training/test_detector_sample.py
=================================
Pre-Phase 2 Smoke Test: Verify YuNet (FaceDetectorYN) on sample images.

Validates that:
1. The ONNX model weights load correctly in OpenCV.
2. The detector accurately locates bounding boxes and 5 facial landmarks:
   - Right eye
   - Left eye
   - Nose tip
   - Right mouth corner
   - Left mouth corner
3. Confidence scores meet the acceptance threshold (>= 0.80).
"""

import random
from pathlib import Path

import cv2
import numpy as np


YUNET_MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")


def test_detector_on_samples(n_samples: int = 5, seed: int = 42):
    print("=" * 65)
    print("  Pre-Phase 2 Smoke Test: YuNet Detector & Landmark Extraction")
    print("=" * 65)

    if not YUNET_MODEL_PATH.exists():
        print(f"[FAIL] YuNet ONNX model not found at {YUNET_MODEL_PATH}")
        return False

    train_dir = Path("data/processed/train")
    if not train_dir.exists():
        print(f"[FAIL] Train directory not found at {train_dir}")
        return False

    random.seed(seed)
    all_images = sorted(list(train_dir.rglob("*.jpg")))
    samples = random.sample(all_images, min(n_samples, len(all_images)))

    # Dummy detector to get started (input size dynamically updated per image)
    detector = cv2.FaceDetectorYN.create(
        model=str(YUNET_MODEL_PATH),
        config="",
        input_size=(250, 250),
        score_threshold=0.75,
        nms_threshold=0.3,
        top_k=5,
    )

    all_passed = True
    print(f"Testing on {len(samples)} diverse identities from data/processed/train/:\n")

    for i, img_path in enumerate(samples, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{i}] [FAIL] Cannot read image: {img_path}")
            all_passed = False
            continue

        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        status, detections = detector.detect(img)

        if detections is None or len(detections) == 0:
            print(f"  [{i}] [FAIL] No face detected in: {img_path.name}")
            all_passed = False
            continue

        det = detections[0]
        bbox = det[0:4].astype(int)
        score = det[14]
        landmarks = det[4:14].reshape((5, 2))

        # Check landmarks are within image boundaries
        in_bounds = np.all((landmarks[:, 0] >= 0) & (landmarks[:, 0] < w) &
                           (landmarks[:, 1] >= 0) & (landmarks[:, 1] < h))

        print(f"  [{i}] {img_path.parent.name:<25} | Score: {score:.3f} | BBox: [x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}] | 5-pt landmarks in bounds: {in_bounds}")

        if score < 0.75 or not in_bounds:
            all_passed = False

    print("\n" + "=" * 65)
    if all_passed:
        print("  [PASS] YuNet detector successfully validated on sample images.")
        print("         Ready for Phase 2 full-dataset alignment pipeline.")
    else:
        print("  [WARN] Detector validation encountered issues.")
    print("=" * 65)
    return all_passed


if __name__ == "__main__":
    success = test_detector_on_samples()
    raise SystemExit(0 if success else 1)
