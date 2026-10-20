"""
training/enroll_identities.py
==============================
Phase 3 - Step 1: Enroll training identities into FAISS vector gallery.

Reads  : data/processed/aligned_train/<IdentityName>/*.jpg
Outputs: data/database/faiss_index.bin
         data/database/gallery_metadata.json

Extracts 128-D L2-normalized embeddings for all aligned training images,
computes identity prototype centroids, and indexes them into FAISS IndexFlatIP.

Usage:
    python training/enroll_identities.py [--data-dir data/processed/aligned_train]
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

from backend.app.vision.embedder import FaceEmbedder
from backend.app.vision.matcher import FaceMatcher


def enroll_training_gallery(
    aligned_train_dir: Path,
    embedder: FaceEmbedder,
    matcher: FaceMatcher,
) -> dict:
    """
    Extract embeddings for each identity's training images and enroll them into FAISS.
    Also computes and enrolls an 'identity prototype' (mean normalized embedding).
    """
    identity_dirs = sorted([d for d in aligned_train_dir.iterdir() if d.is_dir()])
    print(f"Found {len(identity_dirs)} identity directories in {aligned_train_dir}")

    total_images_enrolled = 0
    prototypes_enrolled = 0

    for id_dir in tqdm(identity_dirs, desc="Enrolling identities into FAISS"):
        identity_name = id_dir.name
        img_paths = sorted(id_dir.glob("*.jpg"))
        if not img_paths:
            continue

        id_embeddings = []
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            emb = embedder.embed(img, normalize=True)
            id_embeddings.append(emb)

            # Enroll individual exemplar image into FAISS
            matcher.add(
                identity=identity_name,
                embedding=emb,
                metadata={
                    "identity": identity_name,
                    "filename": img_path.name,
                    "is_prototype": False,
                },
            )
            total_images_enrolled += 1

        # Compute and enroll identity centroid prototype (mean vector)
        if id_embeddings:
            prototype = np.mean(id_embeddings, axis=0)
            norm = np.linalg.norm(prototype)
            if norm > 1e-8:
                prototype = prototype / norm

            matcher.add(
                identity=identity_name,
                embedding=prototype,
                metadata={
                    "identity": identity_name,
                    "filename": "prototype_centroid",
                    "is_prototype": True,
                    "num_exemplars": len(id_embeddings),
                },
            )
            prototypes_enrolled += 1

    return {
        "identities_count": len(identity_dirs),
        "total_images_enrolled": total_images_enrolled,
        "prototypes_enrolled": prototypes_enrolled,
        "total_vectors_in_index": matcher.ntotal,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 - Enroll Training Identities into FAISS Gallery")
    parser.add_argument("--data-dir", type=str, default="data/processed/aligned_train",
                        help="Path to aligned training dataset")
    parser.add_argument("--model-path", type=str, default="models/face_recognition_sface_2021dec.onnx",
                        help="Path to SFace ONNX model")
    parser.add_argument("--db-dir", type=str, default="data/database",
                        help="Output directory for FAISS index and metadata")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    train_dir = project_root / args.data_dir
    model_path = project_root / args.model_path
    db_dir = project_root / args.db_dir

    print("=" * 65)
    print("  Phase 3 - Identity Enrollment into FAISS Vector Gallery")
    print("=" * 65)
    print(f"  Source Aligned Data : {train_dir}")
    print(f"  Embedding Model     : {model_path}")
    print(f"  Database Output Dir : {db_dir}")
    print()

    if not train_dir.exists():
        print(f"ERROR: Train directory not found at {train_dir}")
        print("       Run training/align_dataset.py first.")
        raise SystemExit(1)

    embedder = FaceEmbedder(model_path=model_path)
    matcher = FaceMatcher(dimension=embedder.embedding_dimension)

    stats = enroll_training_gallery(train_dir, embedder, matcher)

    # Save to disk
    index_file = db_dir / "faiss_index.bin"
    metadata_file = db_dir / "gallery_metadata.json"
    matcher.save(index_file, metadata_file)

    print("\n" + "=" * 65)
    print("  Enrollment Complete:")
    print("=" * 65)
    print(f"  Unique Identities Enrolled : {stats['identities_count']}")
    print(f"  Individual Face Exemplars   : {stats['total_images_enrolled']}")
    print(f"  Identity Centroid Prototypes: {stats['prototypes_enrolled']}")
    print(f"  Total Enrolled Vectors      : {stats['total_vectors_in_index']}")
    print(f"  Saved FAISS Index           : {index_file} ({index_file.stat().st_size // 1024} KB)")
    print(f"  Saved Metadata              : {metadata_file} ({metadata_file.stat().st_size // 1024} KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
