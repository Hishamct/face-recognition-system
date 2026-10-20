from fastapi import APIRouter

try:
    from app.api.routes import detection, evaluation, identities, recognition, training
except ImportError:
    from backend.app.api.routes import detection, evaluation, identities, recognition, training

api_router = APIRouter()

api_router.include_router(detection.router)
api_router.include_router(recognition.router)
api_router.include_router(identities.router)
api_router.include_router(evaluation.router)
api_router.include_router(training.router)
