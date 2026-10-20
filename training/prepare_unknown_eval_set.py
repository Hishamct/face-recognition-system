"""
training/prepare_unknown_eval_set.py
====================================
Phase 3 - Evaluation Correction: Prepare Real Un-Enrolled Human Face Datasets.

Extracts two disjoint partitions of real un-enrolled human faces from data/lfw/:
1. data/processed/aligned_unknown_val/ (200 identities)
   - Used to calibrate 1:N gallery-level open-set false acceptance / rejection rates.
2. data/processed/aligned_unknown_test/ (200 identities)
   - Used exclusively for held-out open-set testing (never touched during calibration).

All selected identities are strictly disjoint from the 150 enrolled gallery identities
and disjoint from each other.

Usage:
    python training/prepare_unknown_eval_set.py
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import cv2
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.vision.aligner import FaceAligner
from backend.app.vision.detector import YuNetDetector


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Prepare Real Unknown Identity Sets for Open-Set Evaluation")
    parser.add_argument("--lfw-dir", type=str, default="data/lfw", help="Path to raw uncompressed LFW")
    parser.add_argument("--gallery-meta", type=str, default="data/database/gallery_metadata.json",
                        help="Path to enrolled gallery metadata")
    parser.add_argument("--val-count", type=int, default=200, help="Number of unknown identities for validation")
    parser.add_argument("--test-count", type=int, default=200, help="Number of unknown identities for test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic selection")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    lfw_dir = project_root / args.lfw_dir
    gallery_meta_path = project_root / args.gallery_meta

    if not lfw_dir.exists():
        print(f"ERROR: LFW directory not found at {lfw_dir}")
        raise SystemExit(1)

    # 1. Load enrolled gallery identities to guarantee 100% exclusion
    with open(gallery_meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    gallery_identities = set(meta_data.get("identities", []))
    print(f"Enrolled gallery contains {len(gallery_identities)} unique identities.")

    # 2. Find eligible non-gallery directories
    all_lfw_dirs = sorted([d for d in lfw_dir.iterdir() if d.is_dir()])
    non_gallery_dirs = [d for d in all_lfw_dirs if d.name not in gallery_identities]
    print(f"Found {len(non_gallery_dirs)} non-gallery candidate identity folders.")

    # 3. Deterministically shuffle non-gallery directories
    rng = random.Random(args.seed)
    rng.shuffle(non_gallery_dirs)

    detector = YuNetDetector()
    aligner = FaceAligner()

    val_out_dir = project_root / "data/processed/aligned_unknown_val"
    test_out_dir = project_root / "data/processed/aligned_unknown_test"
    val_out_dir.mkdir(parents=True, exist_ok=True)
    test_out_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []

    # 4. Extract validation unknown set
    val_collected = 0
    candidate_idx = 0

    print(f"\nExtracting {args.val_count} real unknown validation identities...")
    with tqdm(total=args.val_count, desc="Unknown Val Set") as pbar:
        while val_collected < args.val_count and candidate_idx < len(non_gallery_dirs):
            cand_dir = non_gallery_dirs[candidate_idx]
            candidate_idx += 1

            imgs = sorted(cand_dir.glob("*.jpg"))
            if not imgs:
                continue

            src_img_p = imgs[0]
            img = cv2.imread(str(src_img_p))
            if img is None:
                continue

            # Blur check
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 40.0:
                continue

            dets = detector.detect(img)
            if not dets:
                continue

            aligned = aligner.align(img, dets[0].landmarks)
            id_name = cand_dir.name
            target_dir = val_out_dir / id_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"{id_name}_0001.jpg"
            cv2.imwrite(str(target_path), aligned)

            val_collected += 1
            pbar.update(1)

            manifest_records.append({
                "identity": id_name,
                "split": "unknown_val",
                "source_file": str(src_img_p.relative_to(project_root)),
                "aligned_file": str(target_path.relative_to(project_root)),
                "sha256": sha256_file(target_path),
            })

    # 5. Extract test unknown set
    test_collected = 0
    print(f"\nExtracting {args.test_count} real unknown test identities...")
    with tqdm(total=args.test_count, desc="Unknown Test Set") as pbar:
        while test_collected < args.test_count and candidate_idx < len(non_gallery_dirs):
            cand_dir = non_gallery_dirs[candidate_idx]
            candidate_idx += 1

            imgs = sorted(cand_dir.glob("*.jpg"))
            if not imgs:
                continue

            src_img_p = imgs[0]
            img = cv2.imread(str(src_img_p))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 40.0:
                continue

            dets = detector.detect(img)
            if not dets:
                continue

            aligned = aligner.align(img, dets[0].landmarks)
            id_name = cand_dir.name
            target_dir = test_out_dir / id_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"{id_name}_0001.jpg"
            cv2.imwrite(str(target_path), aligned)

            test_collected += 1
            pbar.update(1)

            manifest_records.append({
                "identity": id_name,
                "split": "unknown_test",
                "source_file": str(src_img_p.relative_to(project_root)),
                "aligned_file": str(target_path.relative_to(project_root)),
                "sha256": sha256_file(target_path),
            })

    # 6. Save manifest
    manifest_p = project_root / "reports/unknown_evaluation_manifest.json"
    manifest_p.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump({
            "num_unknown_val": val_collected,
            "num_unknown_test": test_collected,
            "records": manifest_records,
        }, f, indent=2)

    print("\n" + "=" * 65)
    print("  Real Unknown Evaluation Set Summary:")
    print("=" * 65)
    print(f"  Unknown Validation Identities : {val_collected} in {val_out_dir}")
    print(f"  Unknown Test Identities       : {test_collected} in {test_out_dir}")
    print(f"  Manifest Saved                : {manifest_p}")
    print("=" * 65)


if __name__ == "__main__":
    main()
