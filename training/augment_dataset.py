"""
training/augment_dataset.py
============================
Phase 1 - Step 4: Albumentations augmentation pipeline definition and
preview generator with identity-preserving, moderated parameters.

CRITICAL DESIGN RULES:
1. Augmentation is applied ONLY to training data at runtime (in PyTorch DataLoader, Phase 4).
2. Validation (val/) and test (test/) splits are NEVER augmented.
3. Transformed images are NEVER saved into the training dataset folder (no disk pollution).
4. Parameters are conservative so key facial landmarks (eyes, nose, mouth) remain intact.

Output:
    reports/augmentation_preview/
        original/     - Untransformed reference samples (1 folder)
        flipped/      - HorizontalFlip (1 transform)
        brightness/   - RandomBrightnessContrast (1 transform)
        blur/         - MotionBlur (1 transform)
        noise/        - GaussNoise (1 transform)
        rotated/      - Rotate (1 transform)
        dropout/      - CoarseDropout (1 transform)
        hsv/          - HueSaturationValue (1 transform)
        clahe/        - CLAHE (1 transform)
        combined/     - Full runtime pipeline (all transforms combined) (1 folder)
        Total: 10 directories (8 individual transforms + 1 original + 1 combined)

Usage:
    python training/augment_dataset.py [--n-samples 5] [--preview-dir reports/augmentation_preview]
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
    The complete training augmentation pipeline for Phase 4 DataLoader.
    All transformations use conservative bounds to preserve facial identity.
    """
    return A.Compose([
        # Horizontal flip: Natural mirror pose variation
        A.HorizontalFlip(p=0.5),

        # Lighting / ambient exposure variation
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),

        # Subtle sensor motion blur (kept low to avoid unrecoverable feature erasure)
        A.MotionBlur(blur_limit=5, p=0.15),

        # Mild camera sensor noise
        A.GaussNoise(std_range=(0.05, 0.15), p=0.2),

        # Slight head tilt (+/- 10 degrees max; alignment in Phase 2 normalizes roll)
        A.Rotate(limit=10, p=0.3),

        # Mild occlusion (sunglasses rim, mask edge) - small holes so landmarks aren't erased
        A.CoarseDropout(
            num_holes_range=(1, 2),
            hole_height_range=(0.03, 0.08),
            hole_width_range=(0.03, 0.08),
            fill=0,
            p=0.2,
        ),

        # Conservative color shifts (skin tone stability)
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.25),

        # Contrast-limited adaptive histogram equalization (enhances local contrast gently)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.15),
    ])


def get_individual_transforms() -> dict:
    """
    Returns named individual transforms for preview generation.
    Each transform is isolated at p=1.0 so that the visual effect can be clearly audited.
    """
    return {
        "flipped":    A.HorizontalFlip(p=1.0),
        "brightness": A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
        "blur":       A.MotionBlur(blur_limit=5, p=1.0),
        "noise":      A.GaussNoise(std_range=(0.1, 0.2), p=1.0),
        "rotated":    A.Rotate(limit=10, p=1.0),
        "dropout":    A.CoarseDropout(
                          num_holes_range=(2, 3),
                          hole_height_range=(0.04, 0.08),
                          hole_width_range=(0.04, 0.08),
                          fill=0,
                          p=1.0,
                      ),
        "hsv":        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0),
        "clahe":      A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=1.0),
    }


# --- Helpers -----------------------------------------------------------------

def load_sample_images(train_dir: Path, n: int, seed: int = 42) -> list:
    """
    Load n sample images exclusively from the training split.
    Asserts that sample images originate only from data/processed/train.
    """
    assert "train" in str(train_dir).lower(), "Safety assertion: Samples must come ONLY from train split!"
    random.seed(seed)
    all_images = sorted(list(train_dir.rglob("*.jpg")))
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
    import shutil
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating augmentation previews -> {preview_dir}")
    print(f"  Samples: {len(samples)} (drawn exclusively from data/processed/train/)")

    for i, (identity, img_rgb) in enumerate(samples):
        filename = f"sample_{i+1:02d}_{identity}.jpg"

        # 1. Reference folder: original unaugmented image
        save_preview(img_rgb, preview_dir / "original", filename)

        # 2-9. Individual transform folders (8 distinct transforms)
        for transform_name, transform in individual_transforms.items():
            try:
                result = transform(image=img_rgb)
                save_preview(result["image"], preview_dir / transform_name, filename)
            except Exception as e:
                print(f"  [warn] {transform_name} failed on sample {i+1}: {e}")

        # 10. Combined pipeline folder (full composed pipeline)
        try:
            result = full_pipeline(image=img_rgb)
            save_preview(result["image"], preview_dir / "combined", filename)
        except Exception as e:
            print(f"  [warn] combined pipeline failed on sample {i+1}: {e}")

    # Count previews generated
    all_previews = list(preview_dir.rglob("*.jpg"))
    print(f"  [OK] {len(all_previews)} preview images saved to {preview_dir}")

    # Print directory layout
    print(f"\n  Preview folders breakdown (8 individual + 1 original + 1 combined = 10 total):")
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
    print("  Phase 1 - Augmentation Pipeline & Preview Generator")
    print("=" * 60)

    if not train_dir.exists():
        print(f"ERROR: training data not found at {train_dir}")
        print("       Run split_dataset.py first.")
        raise SystemExit(1)

    # Load samples strictly from train_dir
    samples = load_sample_images(train_dir, n=args.n_samples, seed=args.seed)
    if not samples:
        print("ERROR: No .jpg images found in data/processed/train/")
        raise SystemExit(1)

    print(f"  Loaded {len(samples)} sample images from training split.")

    full_pipeline = get_full_pipeline()
    individual_transforms = get_individual_transforms()

    print("\n  Moderated Augmentation Pipeline (for Phase 4 DataLoader):")
    for t in full_pipeline.transforms:
        print(f"    - {t.__class__.__name__}")

    generate_previews(samples, preview_dir, full_pipeline, individual_transforms)

    print("\n" + "=" * 60)
    print("  Step 4 COMPLETE - augmentation pipeline validated")
    print(f"  Inspected previews in {preview_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
