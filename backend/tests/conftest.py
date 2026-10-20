import sys
from pathlib import Path

# Ensure both FaceDetection root and FaceDetection/backend are in sys.path
backend_dir = Path(__file__).resolve().parent.parent
project_root = backend_dir.parent

for p in [str(backend_dir), str(project_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Unify app and backend.app in sys.modules to prevent duplicate SQLAlchemy Declarative Base registration
import app
if "backend.app" not in sys.modules:
    sys.modules["backend.app"] = app

import app.db.models
if "backend.app.db.models" not in sys.modules:
    sys.modules["backend.app.db.models"] = app.db.models

import app.db.session
if "backend.app.db.session" not in sys.modules:
    sys.modules["backend.app.db.session"] = app.db.session

import app.services.vector_store
if "backend.app.services.vector_store" not in sys.modules:
    sys.modules["backend.app.services.vector_store"] = app.services.vector_store

import app.vision.matcher
if "backend.app.vision.matcher" not in sys.modules:
    sys.modules["backend.app.vision.matcher"] = app.vision.matcher

import app.vision.aligner
if "backend.app.vision.aligner" not in sys.modules:
    sys.modules["backend.app.vision.aligner"] = app.vision.aligner

import app.vision.detector
if "backend.app.vision.detector" not in sys.modules:
    sys.modules["backend.app.vision.detector"] = app.vision.detector

import app.vision.embedder
if "backend.app.vision.embedder" not in sys.modules:
    sys.modules["backend.app.vision.embedder"] = app.vision.embedder

import app.vision.classifier
if "backend.app.vision.classifier" not in sys.modules:
    sys.modules["backend.app.vision.classifier"] = app.vision.classifier

import app.vision.video
if "backend.app.vision.video" not in sys.modules:
    sys.modules["backend.app.vision.video"] = app.vision.video
