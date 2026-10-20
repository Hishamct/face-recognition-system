"""
training/augment_dataset.py
============================
Phase 1 - Step 4: Define the Albumentations augmentation pipeline and
generate visual preview images showing each transform applied to a sample.

This script does NOT bake augmented images into disk.
Augmentation is applied at runtime inside the PyTorch DataLoader (Phase 4).

Usage:
    python training/augment_dataset.py [--n-samples 5] [--preview-dir reports/augmentation_preview]

Output:
    reports/augmentation_preview/
        original/     - original sample images
        flipped/      - HorizontalFlip
        brightness/   - RandomBrightnessContrast
        blur/         - MotionBlur
        noise/        - GaussNoise
        rotated/      - Rotate
        dropout/      - CoarseDropout
        hsv/          - HueSaturationValue
        clahe/        - CLAHE
        combined/     - full pipeline (all transforms at once)
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:
    print("ERROR: albumentations is not installed.")
    print("       Run: pip install albumentations")
    raise SystemExit(1)


# --- Augmentation Pipeline ---------------------------------------------------

def get_full_pipeline() -> A.Compose:
    """
    The complete training augmentation pipeline.
    This is the SAME pipeline that will be used inside the Phase 4 DataLoader.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.GaussNoise(p=0.3),
        A.Rotate(limit=15, p=0.4),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15),
            fill=0,
            p=0.3,
        ),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.2),
    ])


def get_individual_transforms() -> dict:
    """
    Returns named individual transforms for preview generation.
    Each is applied at p=1.0 so the preview definitely shows the effect.
    """
    return {
        "flipped":    A.HorizontalFlip(p=1.0),
        "brightness": A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
        "blur":       A.MotionBlur(blur_limit=11, p=1.0),
        "noise":      A.GaussNoise(p=1.0),
        "rotated":    A.Rotate(limit=20, p=1.0),
        "dropout":    A.CoarseDropout(
                          num_holes_range=(2, 4),
                          hole_height_range=(0.1, 0.2),
                          hole_width_range=(0.1, 0.2),
                          fill=0,
                          p=1.0,
                      ),
        "hsv":        A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1.0),
        "clahe":      A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=1.0),
    }


# --- Helpers -----------------------------------------------------------------

def load_sample_images(train_dir: Path, n: int, seed: int = 42) -> list:
    """Load n sample images from the training split, one per identity."""
    random.seed(seed)
    all_images = list(train_dir.rglob("*.jpg"))
    if not all_images:
        return []

    selected = random.sample(all_images, min(n, len(all_images)))
    samples = []
    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            samples.append((img_path.parent.name, img_rgb))
    return samples


def save_preview(img_rgb: np.ndarray, out_dir: Path, filename: str) -> None:
    """Convert RGB -> BGR and save preview image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / filename), img_bgr)


def generate_previews(
    samples: list,
    preview_dir: Path,
    full_pipeline: A.Compose,
    individual_transforms: dict,
) -> None:
    """Generate and save preview images for all transforms."""
    print(f"\nGenerating augmentation previews -> {preview_dir}")
    print(f"  Samples: {len(samples)}")

    for i, (identity, img_rgb) in enumerate(samples):
        filename = f"sample_{i+1:02d}_{identity}.jpg"

        # Save original
        save_preview(img_rgb, preview_dir / "original", filename)

        # Save individual transforms
        for transform_name, transform in individual_transforms.items():
            try:
                result = transform(image=img_rgb)
                save_preview(result["image"], preview_dir / transform_name, filename)
            except Exception as e:
                print(f"  [warn] {transform_name} failed on sample {i+1}: {e}")

        # Save combined full pipeline
        try:
            result = full_pipeline(image=img_rgb)
            save_preview(result["image"], preview_dir / "combined", filename)
        except Exception as e:
            print(f"  [warn] combined pipeline failed on sample {i+1}: {e}")

    # Count previews generated
    all_previews = list(preview_dir.rglob("*.jpg"))
    print(f"  [OK] {len(all_previews)} preview images saved to {preview_dir}")

    # Print directory layout
    print(f"\n  Preview folders:")
    for subfolder in sorted(preview_dir.iterdir()):
        if subfolder.is_dir():
            count = len(list(subfolder.glob("*.jpg")))
            print(f"    {subfolder.name}/   ({count} images)")


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 - Augmentation pipeline and preview generator")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of sample images to use for previews (default: 5)")
    parser.add_argument("--preview-dir", type=str, default="reports/augmentation_preview",
                        help="Output directory for preview images")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root data directory (default: data/)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    train_dir = project_root / args.data_dir / "processed" / "train"
    preview_dir = project_root / args.preview_dir

    print("=" * 60)
    print("  Phase 1 - Augmentation Pipeline & Preview")
    print("=" * 60)

    if not train_dir.exists():
        print(f"ERROR: training data not found at {train_dir}")
        print("       Run split_dataset.py first.")
        raise SystemExit(1)

    # Load samples
    samples = load_sample_images(train_dir, n=args.n_samples, seed=args.seed)
    if not samples:
        print("ERROR: No .jpg images found in data/processed/train/")
        raise SystemExit(1)

    print(f"  Loaded {len(samples)} sample images from training split.")

    # Define transforms
    full_pipeline = get_full_pipeline()
    individual_transforms = get_individual_transforms()

    print("\n  Augmentation pipeline:")
    for t in full_pipeline.transforms:
        print(f"    - {t.__class__.__name__}")

    # Generate previews
    generate_previews(samples, preview_dir, full_pipeline, individual_transforms)

    print("\n" + "=" * 60)
    print("  Step 4 COMPLETE - augmentation pipeline validated")
    print(f"  Open {preview_dir} to inspect results visually")
    print("=" * 60)

    # Print the pipeline in a copy-paste ready format for Phase 4
    print("\n  --- Copy this into your Phase 4 DataLoader ---")
    print("""
  from albumentations import (
      Compose, HorizontalFlip, RandomBrightnessContrast,
      MotionBlur, GaussNoise, Rotate, CoarseDropout,
      HueSaturationValue, CLAHE,
  )

  TRAIN_TRANSFORMS = Compose([
      HorizontalFlip(p=0.5),
      RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
      MotionBlur(blur_limit=7, p=0.2),
      GaussNoise(p=0.3),
      Rotate(limit=15, p=0.4),
      CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.05, 0.15),
                    hole_width_range=(0.05, 0.15), fill=0, p=0.3),
      HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
      CLAHE(clip_limit=4.0, p=0.2),
  ])
    """)


if __name__ == "__main__":
    main()
