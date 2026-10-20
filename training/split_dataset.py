"""
training/split_dataset.py
==========================
Phase 1 - Step 3: Stratified 70/15/15 train/val/test 3-way split.

Reads  : data/raw/<IdentityName>/*.jpg
Outputs: data/processed/train/<IdentityName>/*.jpg
         data/processed/val/<IdentityName>/*.jpg
         data/processed/test/<IdentityName>/*.jpg
Updates: reports/dataset_summary.csv  (adds train_count, val_count, test_count)

Usage:
    python training/split_dataset.py [--train-ratio 0.70] [--val-ratio 0.15] [--test-ratio 0.15] [--seed 42]
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

from tqdm import tqdm


# --- Helpers -----------------------------------------------------------------

def load_dataset_summary(csv_path: Path) -> list:
    """Load existing dataset_summary.csv rows."""
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_dataset_summary(rows: list, csv_path: Path) -> None:
    """Overwrite dataset_summary.csv with updated train/val/test counts."""
    fieldnames = ["identity", "total_images", "train_count", "val_count", "test_count"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def partition_images(images: list, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> tuple:
    """
    Deterministically partition images for a single identity into (train, val, test).
    Ensures that for any identity with >= 3 images, every partition receives at least 1 image.
    """
    rng = random.Random(seed)
    shuffled = sorted(images)  # Canonical sort first for exact cross-platform reproducibility
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n < 3:
        raise ValueError(f"Identity has only {n} images; at least 3 required for 3-way split.")

    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_test - n_val

    # Safeguard: ensure train has at least 1 image
    while n_train < 1 and n_val > 1:
        n_val -= 1
        n_train += 1
    while n_train < 1 and n_test > 1:
        n_test -= 1
        n_train += 1

    test_imgs = shuffled[:n_test]
    val_imgs = shuffled[n_test : n_test + n_val]
    train_imgs = shuffled[n_test + n_val :]

    return train_imgs, val_imgs, test_imgs


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 - Stratified 3-way train/val/test split")
    parser.add_argument("--train-ratio", type=float, default=0.70,
                        help="Fraction for training set (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction for validation/calibration set (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="Fraction for final held-out test set (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root data directory (default: data/)")
    args = parser.parse_args()

    # Normalize ratios if sum != 1.0
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    train_r = args.train_ratio / total_ratio
    val_r = args.val_ratio / total_ratio
    test_r = args.test_ratio / total_ratio

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"
    reports_dir = project_root / "reports"
    csv_path = reports_dir / "dataset_summary.csv"

    print("=" * 60)
    print("  Phase 1 - Stratified 3-Way Train / Val / Test Split")
    print("=" * 60)
    print(f"  Target ratios : Train {train_r*100:.1f}% | Val {val_r*100:.1f}% | Test {test_r*100:.1f}%")
    print(f"  Random seed   : {args.seed}")
    print(f"  Source        : {raw_dir}")
    print(f"  Output        : {processed_dir}")
    print()

    if not raw_dir.exists():
        print(f"ERROR: raw_dir does not exist: {raw_dir}")
        print("       Run prepare_dataset.py first.")
        raise SystemExit(1)

    # Clean existing processed train/val/test directories before splitting
    for p_dir in [train_dir, val_dir, test_dir]:
        if p_dir.exists():
            shutil.rmtree(p_dir)
        p_dir.mkdir(parents=True, exist_ok=True)

    # Discover identity folders
    identity_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not identity_dirs:
        print("ERROR: No identity folders found in data/raw/")
        raise SystemExit(1)

    print(f"Found {len(identity_dirs)} identity folders to partition.\n")

    existing_rows = load_dataset_summary(csv_path)
    row_map = {r["identity"]: r for r in existing_rows}

    train_total = 0
    val_total = 0
    test_total = 0
    skipped = 0

    for idx, identity_dir in enumerate(tqdm(identity_dirs, desc="Splitting identities")):
        identity_name = identity_dir.name
        images = sorted(identity_dir.glob("*.jpg"))

        if len(images) < 3:
            skipped += 1
            continue

        # Per-identity deterministic seed derived from base seed + index
        id_seed = args.seed + idx
        train_imgs, val_imgs, test_imgs = partition_images(
            images, train_r, val_r, test_r, seed=id_seed
        )

        (train_dir / identity_name).mkdir(parents=True, exist_ok=True)
        (val_dir / identity_name).mkdir(parents=True, exist_ok=True)
        (test_dir / identity_name).mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy2(str(img), str(train_dir / identity_name / img.name))
        for img in val_imgs:
            shutil.copy2(str(img), str(val_dir / identity_name / img.name))
        for img in test_imgs:
            shutil.copy2(str(img), str(test_dir / identity_name / img.name))

        train_c = len(train_imgs)
        val_c = len(val_imgs)
        test_c = len(test_imgs)

        train_total += train_c
        val_total += val_c
        test_total += test_c

        row_map[identity_name] = {
            "identity": identity_name,
            "total_images": train_c + val_c + test_c,
            "train_count": train_c,
            "val_count": val_c,
            "test_count": test_c,
        }

    reports_dir.mkdir(exist_ok=True)
    summary_rows = [row_map[k] for k in sorted(row_map.keys()) if k in row_map]
    write_dataset_summary(summary_rows, csv_path)

    print("\n[OK] 3-Way Split complete.")
    print(f"  Identities processed : {len(identity_dirs) - skipped}")
    print(f"  Skipped (< 3 images) : {skipped}")
    print(f"  Total train images   : {train_total} ({train_total/(train_total+val_total+test_total)*100:.1f}%)")
    print(f"  Total val images     : {val_total} ({val_total/(train_total+val_total+test_total)*100:.1f}%)")
    print(f"  Total test images    : {test_total} ({test_total/(train_total+val_total+test_total)*100:.1f}%)")
    print(f"  Total images         : {train_total + val_total + test_total}")
    print(f"  Updated summary      : {csv_path}")

    # Verification counts directly from disk
    actual_train = sum(1 for _ in train_dir.rglob("*.jpg"))
    actual_val = sum(1 for _ in val_dir.rglob("*.jpg"))
    actual_test = sum(1 for _ in test_dir.rglob("*.jpg"))
    print("\n  Verification (disk count):")
    print(f"    train/*.jpg : {actual_train}")
    print(f"    val/*.jpg   : {actual_val}")
    print(f"    test/*.jpg  : {actual_test}")

    print("\n" + "=" * 60)
    print("  Step 3 COMPLETE - 3-Way Split validated")
    print("=" * 60)


if __name__ == "__main__":
    main()
