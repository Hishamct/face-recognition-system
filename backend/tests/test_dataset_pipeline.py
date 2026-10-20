"""
backend/tests/test_dataset_pipeline.py
=======================================
Automated test suite verifying Phase 1 dataset pipeline, 3-way split,
augmentation stability, and zero data leakage.
"""

from pathlib import Path
import csv
import hashlib
import cv2
import numpy as np
import pytest

from training.augment_dataset import get_full_pipeline, get_individual_transforms


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def test_manifest_exists_and_valid(repo_root: Path):
    manifest_path = repo_root / "reports" / "dataset_summary.csv"
    assert manifest_path.exists(), "Manifest reports/dataset_summary.csv must exist"

    with open(manifest_path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) >= 100, f"Expected at least 100 identities, found {len(reader)}"
    for row in reader:
        assert int(row["train_count"]) >= 1, f"Identity {row['identity']} must have train images"
        assert int(row["val_count"]) >= 1, f"Identity {row['identity']} must have val images"
        assert int(row["test_count"]) >= 1, f"Identity {row['identity']} must have test images"


def test_zero_leakage_hashes(repo_root: Path):
    train_dir = repo_root / "data" / "processed" / "train"
    val_dir = repo_root / "data" / "processed" / "val"
    test_dir = repo_root / "data" / "processed" / "test"

    if not train_dir.exists():
        pytest.skip("Processed dataset not generated yet on disk")

    def get_hashes(directory: Path, limit: int = 200) -> set:
        hashes = set()
        for p in list(directory.rglob("*.jpg"))[:limit]:
            with open(p, "rb") as f:
                hashes.add(hashlib.sha256(f.read()).hexdigest())
        return hashes

    train_h = get_hashes(train_dir)
    val_h = get_hashes(val_dir)
    test_h = get_hashes(test_dir)

    assert len(train_h & val_h) == 0, "Train and Val must have 0 identical image hashes"
    assert len(train_h & test_h) == 0, "Train and Test must have 0 identical image hashes"
    assert len(val_h & test_h) == 0, "Val and Test must have 0 identical image hashes"


def test_augmentation_pipeline_shape_preservation():
    pipeline = get_full_pipeline()
    dummy_face = np.full((112, 112, 3), 128, dtype=np.uint8)

    augmented = pipeline(image=dummy_face)["image"]
    assert augmented.shape == (112, 112, 3), "Augmentation must preserve image dimensions"
    assert augmented.dtype == np.uint8, "Augmentation must maintain uint8 dtype"


def test_individual_transforms_work():
    transforms = get_individual_transforms()
    dummy_face = np.full((112, 112, 3), 128, dtype=np.uint8)

    for name, t in transforms.items():
        res = t(image=dummy_face)["image"]
        assert res.shape == (112, 112, 3), f"Transform {name} changed image dimensions"
