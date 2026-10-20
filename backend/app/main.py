import sys
from pathlib import Path

# Ensure backend and project root directories are in sys.path
backend_dir = Path(__file__).resolve().parent.parent
project_root = backend_dir.parent
for d in (str(backend_dir), str(project_root)):
    if d not in sys.path:
        sys.path.insert(0, d)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.api.api import api_router
    from app.core.config import settings
    from app.db.session import verify_db_connection
except ImportError:
    from backend.app.api.api import api_router
    from backend.app.core.config import settings
    from backend.app.db.session import verify_db_connection

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Industry-grade Face Detection, 5-Point Landmark Alignment & ArcFace Biometrics Recognition Platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API V1 Router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "status": "online",
        "docs_url": "/docs",
        "health_check": f"{settings.API_V1_PREFIX}/health"
    }


@app.get("/health")
def health():
    db_check = verify_db_connection()
    return {
        "status": "ok",
        "environment": settings.ENVIRONMENT,
        "database": db_check.get("status", "unknown")
    }


@app.get(f"{settings.API_V1_PREFIX}/health")
def api_health():
    db_check = verify_db_connection()
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT,
        "database": db_check.get("status", "unknown"),
        "database_connected": db_check.get("status") == "connected"
    }
