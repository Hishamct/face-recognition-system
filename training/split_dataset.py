"""
training/split_dataset.py
==========================
Phase 1 - Step 3: Stratified 80/20 train/test split.

Reads  : data/raw/<IdentityName>/*.jpg
Outputs: data/processed/train/<IdentityName>/*.jpg
         data/processed/test/<IdentityName>/*.jpg
Updates: reports/dataset_summary.csv  (adds train_count, test_count columns)

Usage:
    python training/split_dataset.py [--test-size 0.2] [--seed 42]
"""

import argparse
import csv
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm


# --- Helpers -----------------------------------------------------------------

def load_dataset_summary(csv_path: Path) -> list:
    """Load existing dataset_summary.csv rows."""
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_dataset_summary(rows: list, csv_path: Path) -> None:
    """Overwrite dataset_summary.csv with updated train/test counts."""
    fieldnames = ["identity", "total_images", "train_count", "test_count"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 - Stratified 80/20 train/test split")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of images to put in test set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root data directory (default: data/)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    train_dir = processed_dir / "train"
    test_dir = processed_dir / "test"
    reports_dir = project_root / "reports"
    csv_path = reports_dir / "dataset_summary.csv"

    print("=" * 60)
    print("  Phase 1 - Stratified Train/Test Split")
    print("=" * 60)
    print(f"  Test size  : {args.test_size * 100:.0f}%")
    print(f"  Random seed: {args.seed}")
    print(f"  Source     : {raw_dir}")
    print(f"  Output     : {processed_dir}")
    print()

    if not raw_dir.exists():
        print(f"ERROR: raw_dir does not exist: {raw_dir}")
        print("       Run prepare_dataset.py first.")
        raise SystemExit(1)

    # Discover identity folders
    identity_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not identity_dirs:
        print("ERROR: No identity folders found in data/raw/")
        raise SystemExit(1)

    print(f"Found {len(identity_dirs)} identity folders to split.\n")

    # Load existing CSV rows for update
    existing_rows = load_dataset_summary(csv_path)
    row_map = {r["identity"]: r for r in existing_rows}

    train_total = 0
    test_total = 0
    skipped = 0

    for identity_dir in tqdm(identity_dirs, desc="Splitting identities"):
        identity_name = identity_dir.name
        images = sorted(identity_dir.glob("*.jpg"))

        if len(images) < 2:
            # Cannot split an identity with only 1 image
            skipped += 1
            continue

        # Stratified split: at minimum 1 image in test set
        test_frac = args.test_size
        if len(images) < 5:
            # For very small identities (2-4 images) always put 1 in test
            n_test = 1
            train_imgs = images[:-1]
            test_imgs = images[-1:]
        else:
            train_imgs, test_imgs = train_test_split(
                images,
                test_size=test_frac,
                random_state=args.seed,
                shuffle=True,
            )

        # Create destination folders
        (train_dir / identity_name).mkdir(parents=True, exist_ok=True)
        (test_dir / identity_name).mkdir(parents=True, exist_ok=True)

        # Copy images
        for img in train_imgs:
            shutil.copy2(str(img), str(train_dir / identity_name / img.name))
        for img in test_imgs:
            shutil.copy2(str(img), str(test_dir / identity_name / img.name))

        train_count = len(list(train_imgs))
        test_count = len(list(test_imgs))
        train_total += train_count
        test_total += test_count

        # Update CSV row
        if identity_name in row_map:
            row_map[identity_name]["train_count"] = train_count
            row_map[identity_name]["test_count"] = test_count
        else:
            row_map[identity_name] = {
                "identity": identity_name,
                "total_images": train_count + test_count,
                "train_count": train_count,
                "test_count": test_count,
            }

    # Save updated summary
    reports_dir.mkdir(exist_ok=True)
    write_dataset_summary(list(row_map.values()), csv_path)

    print("\n[OK] Split complete.")
    print(f"  Identities processed : {len(identity_dirs) - skipped}")
    print(f"  Skipped (too few)    : {skipped}")
    print(f"  Total train images   : {train_total}")
    print(f"  Total test images    : {test_total}")
    print(f"  Updated summary      : {csv_path}")

    # Quick verification - count actual files on disk
    actual_train = sum(1 for _ in train_dir.rglob("*.jpg"))
    actual_test = sum(1 for _ in test_dir.rglob("*.jpg"))
    print("\n  Verification (disk count):")
    print(f"    train/*.jpg : {actual_train}")
    print(f"    test/*.jpg  : {actual_test}")

    print("\n" + "=" * 60)
    print("  Step 3 COMPLETE - ready for augment_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
