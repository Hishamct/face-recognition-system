from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

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
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    return {
        "status": "ok",
        "environment": settings.ENVIRONMENT
    }


@app.get(f"{settings.API_V1_PREFIX}/health")
def api_health():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT
    }
