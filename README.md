# Face Detection, Recognition & Biometrics Intelligence Platform

An industry-grade, end-to-end computer vision platform for real-time face detection, canonical 5-point landmark alignment, ArcFace feature embedding extraction, and sub-millisecond vector similarity search using FAISS and PostgreSQL.

---

## 📌 Architecture & Technology Stack

- **Face Detection & Alignment**: SCRFD / RetinaFace with 5-point facial landmark regression and affine transformation to canonical $112 \times 112$ px crops.
- **Feature Extraction**: Deep Face Recognition with ArcFace (Additive Angular Margin Loss) producing 512-D L2-normalized embeddings.
- **Vector Search Engine**: FAISS (`IndexFlatIP` / `IndexHNSWFlat`) for sub-millisecond similarity matching against gallery identities.
- **Relational Metadata Store**: PostgreSQL (Identity profiles, detection logs, enrollment records).
- **Backend Service**: FastAPI, Uvicorn, SQLAlchemy, Pydantic, and OpenCV.
- **Frontend Dashboard**: React, TypeScript, Vite, HTML Canvas overlays, and WebRTC / WebSocket streaming.

---

## 🗂️ Project Structure

```text
FaceDetection/
├── .gitignore
├── .env.example
├── README.md
├── PROJECT_BLUEPRINT.md
├── backend/
│   ├── app/
│   │   ├── api/routes/        # REST and WebSocket endpoints
│   │   ├── core/              # App config, database session, security
│   │   ├── db/                # SQLAlchemy models & migrations
│   │   ├── schemas/           # Pydantic validation schemas
│   │   ├── services/          # Business logic and services
│   │   └── vision/            # Face detection, alignment, embeddings, FAISS
│   ├── tests/                 # Unit & integration tests
│   └── requirements.txt       # Python dependencies
├── frontend/                  # React + TypeScript + Vite UI dashboard
├── training/                  # Dataset preparation & supervised training experiments
├── configs/                   # System YAML configuration files
├── data/                      # Local datasets (ignored by git)
│   ├── raw/
│   ├── processed/
│   └── database/
├── models/                    # Pretrained & exported ONNX/PyTorch model weights (ignored by git)
├── reports/                   # Performance charts, confusion matrices, ROC-AUC curves
└── docs/                      # Architectural diagrams & specifications
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.12+
- PostgreSQL 16+
- Node.js 18+ (for frontend dashboard)
- Git

### 2. Environment Configuration
Copy the example environment file and configure your credentials:
```bash
cp .env.example .env
```

### 3. Backend Setup
```bash
cd backend
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies:
pip install -r requirements.txt
```

### 4. Running the Backend Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Swagger API docs: `http://127.0.0.1:8000/docs`
- Health check endpoint: `http://127.0.0.1:8000/api/v1/health`

---

## 📜 Development Roadmap
- [x] **Phase 0**: Environment Setup, Git Repository, PostgreSQL & Health Verification
- [ ] **Phase 1**: Dataset Preparation, Cleaning & Canonical Affine Alignment
- [ ] **Phase 2**: SOTA Face Detection & 5-Point Landmark Extraction (SCRFD / RetinaFace)
- [ ] **Phase 3**: ArcFace Feature Extraction & FAISS Vector Matching
- [ ] **Phase 4**: Supervised Classifier Training & Metric Evaluation (ROC, PR, Confusion Matrix)
- [ ] **Phase 5**: Production FastAPI REST & WebSocket Backend
- [ ] **Phase 6**: Modern React + TypeScript Dashboard & Canvas HUD Overlays
- [ ] **Phase 7**: Real-Time Video Stream & Live Webcam Inference
- [ ] **Phase 8**: Production Polish, ByteTrack Tracking, & Anti-Spoofing (Liveness Check)
