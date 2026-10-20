"""
training/prepare_dataset.py
============================
Phase 1 - Step 1 & 2: Download LFW dataset, organize into per-identity
folders, and run image quality verification.

Usage:
    python training/prepare_dataset.py [--min-images N] [--data-dir PATH]

Defaults:
    --min-images 10     Keep only identities with >= 10 images
    --data-dir   data/  Root data directory
"""

import argparse
import csv
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# --- Constants ---------------------------------------------------------------
# Primary + mirror URLs tried in order until one succeeds
LFW_URLS = [
    "http://vis-www.cs.umass.edu/lfw/lfw.tgz",                 # UMass primary (may be unreachable)
    "https://ndownloader.figshare.com/files/5976018",          # Figshare mirror
]
LFW_ARCHIVE = "lfw.tgz"
MIN_RESOLUTION = 60          # pixels - each dimension must be >= this
BLUR_THRESHOLD = 50.0        # Laplacian variance below this = too blurry

# Note: OpenCV 5 removed CascadeClassifier (Haar cascade).
# Phase 1 quality check uses resolution + blur only.
# The real face detection (SCRFD) is done in Phase 2.


# --- Helpers -----------------------------------------------------------------

def download_lfw(dest_dir: Path, archive_path: Path) -> None:
    """Download LFW .tgz archive, trying each mirror URL until one succeeds."""
    if archive_path.exists():
        print("[skip] Archive already present: {}".format(archive_path))
        return

    print("File size: ~175 MB - this may take a few minutes.\n")

    def _progress(block_num, block_size, total_size):
        if total_size <= 0:
            sys.stdout.write("\r  Downloaded: {} MB".format(block_num * block_size // 1024 // 1024))
        else:
            pct = min(block_num * block_size / total_size * 100, 100)
            bar = int(pct / 2)
            sys.stdout.write("\r  [{}{}] {:.1f}%".format("=" * bar, " " * (50 - bar), pct))
        sys.stdout.flush()

    last_error = None
    for url in LFW_URLS:
        try:
            print("Trying: {}".format(url))
            tmp_path = archive_path.with_suffix(".tmp")
            urllib.request.urlretrieve(url, str(tmp_path), reporthook=_progress)
            print("")
            tmp_path.rename(archive_path)
            print("Download complete.")
            return
        except Exception as e:
            last_error = e
            print("\n  [fail] {}".format(e))
            if tmp_path.exists():
                tmp_path.unlink()
            continue

    raise RuntimeError(
        "All LFW mirror URLs failed. Last error: {}\n"
        "Please download lfw.tgz manually and place it in the data/ folder.\n"
        "Download from: https://ndownloader.figshare.com/files/5976018".format(last_error)
    )


def extract_lfw(archive_path: Path, dest_dir: Path) -> Path:
    """Extract the .tgz archive and return the extracted lfw/ path."""
    lfw_extracted = dest_dir / "lfw"
    if lfw_extracted.exists():
        print("[skip] Already extracted at: {}".format(lfw_extracted))
        return lfw_extracted
    print("Extracting {} -> {} ...".format(archive_path, dest_dir))
    with tarfile.open(str(archive_path), "r:gz") as tar:
        tar.extractall(str(dest_dir), filter="data")
    print("Extraction complete.")
    return lfw_extracted


def blur_score(img_gray: np.ndarray) -> float:
    """Laplacian variance - lower = blurrier."""
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def check_image_quality(img_path: Path) -> tuple:
    """
    Returns (passes: bool, reason: str).
    Checks: minimum resolution and blur.
    Note: Face detection is deferred to Phase 2 (SCRFD detector).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False, "unreadable"

    h, w = img.shape[:2]

    # Resolution check
    if h < MIN_RESOLUTION or w < MIN_RESOLUTION:
        return False, "low_resolution"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur check (Laplacian variance)
    if blur_score(gray) < BLUR_THRESHOLD:
        return False, "blurry"

    return True, "ok"


def organize_and_filter(
    lfw_dir: Path,
    raw_dir: Path,
    min_images: int,
) -> dict:
    """
    Copy images from lfw/ into data/raw/<IdentityName>/ applying:
      - minimum image count filter per identity
      - per-image quality checks

    Returns dict: identity_name -> accepted_image_count
    """
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    identity_dirs = sorted([d for d in lfw_dir.iterdir() if d.is_dir()])
    print("\nFound {} identities in LFW archive.".format(len(identity_dirs)))

    # First pass: count raw images per identity (before quality filter)
    eligible = [d for d in identity_dirs if len(list(d.glob("*.jpg"))) >= min_images]
    print("Identities with >= {} images: {}".format(min_images, len(eligible)))

    rejection_reasons = {
        "low_resolution": 0,
        "blurry": 0,
        "no_face": 0,
        "multiple_faces": 0,
        "unreadable": 0,
    }
    identity_counts = {}

    for identity_dir in tqdm(eligible, desc="Organizing & filtering identities"):
        identity_name = identity_dir.name
        out_dir = raw_dir / identity_name
        out_dir.mkdir(exist_ok=True)

        accepted = 0
        for img_path in sorted(identity_dir.glob("*.jpg")):
            passes, reason = check_image_quality(img_path)
            if passes:
                shutil.copy2(str(img_path), str(out_dir / img_path.name))
                accepted += 1
            else:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        if accepted < min_images:
            # If usable count drops below min_images after quality filter, purge identity
            shutil.rmtree(out_dir)
            rejection_reasons["below_min_usable"] = rejection_reasons.get("below_min_usable", 0) + 1
        else:
            identity_counts[identity_name] = accepted

    total_rejected = sum(rejection_reasons.values())
    print("\n[OK] Quality filter complete.")
    print("  Accepted identities (>= {} usable images): {}".format(min_images, len(identity_counts)))
    print("  Identities purged (< {} usable images)   : {}".format(min_images, rejection_reasons.get("below_min_usable", 0)))
    print("  Total rejected individual images        : {}".format(total_rejected - rejection_reasons.get("below_min_usable", 0)))
    for reason, count in rejection_reasons.items():
        if count > 0:
            print("    - {}: {}".format(reason, count))


    return identity_counts


def write_summary_csv(identity_counts: dict, reports_dir: Path) -> None:
    """Write initial dataset_summary.csv with identity name and total image count."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "dataset_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["identity", "total_images", "train_count", "test_count"])
        writer.writeheader()
        for identity, count in sorted(identity_counts.items()):
            writer.writerow({
                "identity": identity,
                "total_images": count,
                "train_count": "",
                "test_count": "",
            })
    total_images = sum(identity_counts.values())
    print("\n[OK] Dataset summary written: {}".format(csv_path))
    print("  Total identities: {}".format(len(identity_counts)))
    print("  Total images    : {}".format(total_images))


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 - Download and prepare LFW dataset")
    parser.add_argument("--min-images", type=int, default=10,
                        help="Minimum images per identity to include (default: 10)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root data directory (default: data/)")
    args = parser.parse_args()

    # Resolve paths relative to project root (script is inside training/)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    raw_dir = data_dir / "raw"
    archive_path = data_dir / LFW_ARCHIVE
    reports_dir = project_root / "reports"

    print("=" * 60)
    print("  Phase 1 - LFW Dataset Preparation")
    print("=" * 60)
    print("  Project root : {}".format(project_root))
    print("  Data dir     : {}".format(data_dir))
    print("  Min images   : {}".format(args.min_images))
    print()

    # Step 1: Download
    data_dir.mkdir(parents=True, exist_ok=True)
    download_lfw(data_dir, archive_path)

    # Step 2: Extract
    lfw_dir = extract_lfw(archive_path, data_dir)

    # Step 3: Organize + quality filter -> data/raw/
    identity_counts = organize_and_filter(lfw_dir, raw_dir, args.min_images)

    # Step 4: Write initial summary CSV
    write_summary_csv(identity_counts, reports_dir)

    print("\n" + "=" * 60)
    print("  Step 1 & 2 COMPLETE - ready for split_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
