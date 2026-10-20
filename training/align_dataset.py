"""
training/align_dataset.py
==========================
Phase 2 - Full Dataset Face Detection & 5-Point Similarity Alignment.

Processes all images in:
    data/processed/train/ -> data/processed/aligned_train/
    data/processed/val/   -> data/processed/aligned_val/
    data/processed/test/  -> data/processed/aligned_test/

Each face is:
1. Detected via YuNet (FaceDetectorYN).
2. Landmark-aligned into canonical 112x112 coordinates via FaceAligner.
3. Stored into identical identity subfolder hierarchies.

Also generates visual debugging samples and a side-by-side comparison in:
    reports/detection_samples/

Usage:
    python training/align_dataset.py [--model-path models/face_detection_yunet_2023mar.onnx]
"""

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.vision.aligner import FaceAligner
from backend.app.vision.detector import YuNetDetector


def process_partition(
    src_dir: Path,
    dst_dir: Path,
    detector: YuNetDetector,
    aligner: FaceAligner,
    desc: str,
) -> dict:
    """Detect and align all faces in a dataset partition."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    identity_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()])

    stats = {
        "total": 0,
        "aligned": 0,
        "fallback": 0,
        "no_face": 0,
    }

    for id_dir in tqdm(identity_dirs, desc=desc):
        out_id_dir = dst_dir / id_dir.name
        out_id_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(id_dir.glob("*.jpg"))
        for img_path in images:
            stats["total"] += 1
            img = cv2.imread(str(img_path))
            if img is None:
                stats["no_face"] += 1
                continue

            detections = detector.detect(img)
            if not detections:
                # Detector found no face with high confidence; fallback to center-crop resize
                h, w = img.shape[:2]
                fallback_crop = cv2.resize(img, aligner.target_size)
                cv2.imwrite(str(out_id_dir / img_path.name), fallback_crop)
                stats["fallback"] += 1
                continue

            # Pick top detection (highest confidence)
            best_det = detections[0]
            aligned_face = aligner.align(
                img,
                best_det.landmarks,
                fallback_bbox=best_det.bbox,
            )

            cv2.imwrite(str(out_id_dir / img_path.name), aligned_face)
            stats["aligned"] += 1

    return stats


def generate_visual_samples(
    sample_images: list,
    detector: YuNetDetector,
    aligner: FaceAligner,
    output_dir: Path,
) -> None:
    """Generate side-by-side detection and alignment inspection images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating visual inspection samples -> {output_dir}")

    comparison_panels = []

    for idx, img_path in enumerate(sample_images[:6], start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        detections = detector.detect(img)
        if not detections:
            continue

        det = detections[0]
        # 1. Annotated full image (bbox + landmarks)
        annotated = detector.draw_detections(img, [det])

        # 2. Aligned 112x112 crop
        aligned = aligner.align(img, det.landmarks, fallback_bbox=det.bbox)

        # Save individual inspection files
        id_name = img_path.parent.name
        base_name = f"sample_{idx:02d}_{id_name}"
        cv2.imwrite(str(output_dir / f"{base_name}_annotated.jpg"), annotated)
        cv2.imwrite(str(output_dir / f"{base_name}_aligned_112.jpg"), aligned)

        # Prepare side-by-side panel: scale annotated to 250x250, aligned to 250x250 for easy viewing
        ann_resized = cv2.resize(annotated, (250, 250))
        aligned_zoomed = cv2.resize(aligned, (250, 250))
        # Draw target landmark guide dots on zoomed aligned face for verification
        aligned_with_guide = aligned_zoomed.copy()
        for (gx, gy) in aligner.reference_landmarks:
            cv2.circle(aligned_with_guide, (int(gx * 250 / 112), int(gy * 250 / 112)), 4, (0, 255, 0), -1)

        panel = np.hstack([ann_resized, aligned_with_guide])
        # Add label
        cv2.putText(panel, "Detected (YuNet)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(panel, "Aligned 112x112 (Canonical)", (260, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        comparison_panels.append(panel)

    if comparison_panels:
        # Stack up to 3 panels vertically into a comparison grid
        grid = np.vstack(comparison_panels[:3])
        cv2.imwrite(str(output_dir / "alignment_comparison_grid.jpg"), grid)
        print(f"  [OK] Saved comparison grid: {output_dir / 'alignment_comparison_grid.jpg'}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 - Face Detection & 5-Point Alignment")
    parser.add_argument("--model-path", type=str, default="models/face_detection_yunet_2023mar.onnx",
                        help="Path to YuNet ONNX model")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Root processed data directory")
    parser.add_argument("--reports-dir", type=str, default="reports/detection_samples",
                        help="Directory for visual debugging samples")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    reports_dir = project_root / args.reports_dir
    model_path = project_root / args.model_path

    print("=" * 65)
    print("  Phase 2 - Face Detection & 5-Point Affine Alignment Pipeline")
    print("=" * 65)
    print(f"  YuNet Model : {model_path}")
    print(f"  Input Data  : {data_dir}")
    print(f"  Visual Out  : {reports_dir}")
    print()

    detector = YuNetDetector(model_path=model_path, score_threshold=0.75, nms_threshold=0.3)
    aligner = FaceAligner(target_size=(112, 112))

    # 1. Process Train split
    train_src = data_dir / "train"
    train_dst = data_dir / "aligned_train"
    train_stats = process_partition(train_src, train_dst, detector, aligner, "Aligning Train split")

    # 2. Process Val split
    val_src = data_dir / "val"
    val_dst = data_dir / "aligned_val"
    val_stats = process_partition(val_src, val_dst, detector, aligner, "Aligning Val split  ")

    # 3. Process Test split
    test_src = data_dir / "test"
    test_dst = data_dir / "aligned_test"
    test_stats = process_partition(test_src, test_dst, detector, aligner, "Aligning Test split ")

    # 4. Generate Visual Samples
    sample_images = sorted(list(train_src.rglob("*.jpg")))
    generate_visual_samples(sample_images, detector, aligner, reports_dir)

    print("\n" + "=" * 65)
    print("  Phase 2 Alignment Results Summary:")
    print("=" * 65)
    print(f"  Train : {train_stats['aligned']} aligned, {train_stats['fallback']} fallback of {train_stats['total']} total")
    print(f"  Val   : {val_stats['aligned']} aligned, {val_stats['fallback']} fallback of {val_stats['total']} total")
    print(f"  Test  : {test_stats['aligned']} aligned, {test_stats['fallback']} fallback of {test_stats['total']} total")
    total_aligned = train_stats['aligned'] + val_stats['aligned'] + test_stats['aligned']
    total_imgs = train_stats['total'] + val_stats['total'] + test_stats['total']
    success_rate = (total_aligned / total_imgs) * 100 if total_imgs > 0 else 0
    print(f"  Overall Alignment Success Rate: {success_rate:.2f}% ({total_aligned}/{total_imgs})")
    print("=" * 65)


if __name__ == "__main__":
    main()
