"""
backend/app/api/routes/video.py
===============================
WebSocket and REST endpoints for live video streaming and video file processing.
"""

import base64
import os
import tempfile
import time
from typing import List, Optional

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel

try:
    from app.services.recognition_service import recognition_service
    from app.vision.video import VideoProcessor
except ImportError:
    from backend.app.services.recognition_service import recognition_service
    from backend.app.vision.video import VideoProcessor

router = APIRouter(prefix="/video", tags=["Video Processing & Streaming"])


class VideoSummaryResponse(BaseModel):
    success: bool
    filename: str
    total_frames_processed: int
    duration_seconds: float
    average_fps: float
    average_latency_ms: float
    unique_identities_detected: List[str]
    timeline_sample: List[dict]


@router.websocket("/stream")
async def websocket_video_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live webcam face recognition.
    Accepts raw binary JPEG frame bytes or JSON payloads with base64 encoded images.
    Sends back structured detection boxes, landmarks, matched identities, and latency telemetry.
    """
    await websocket.accept()
    video_proc = VideoProcessor(
        detector=recognition_service.detector,
        aligner=recognition_service.aligner,
        embedder=recognition_service.embedder,
        classifier=recognition_service.classifier,
        threshold=recognition_service.DEFAULT_CLASSIFIER_THRESHOLD,
    )

    try:
        while True:
            # Receive frame data from client
            message = await websocket.receive()

            frame_bytes: Optional[bytes] = None
            threshold_override: Optional[float] = None

            if "bytes" in message and message["bytes"]:
                frame_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                text_data = message["text"]
                # Can be a raw base64 string or a JSON object
                if text_data.startswith("{"):
                    import json
                    try:
                        payload = json.loads(text_data)
                        b64_str = payload.get("data") or payload.get("frame") or ""
                        threshold_override = payload.get("threshold")
                        if "," in b64_str:
                            b64_str = b64_str.split(",", 1)[1]
                        frame_bytes = base64.b64decode(b64_str)
                    except Exception:
                        frame_bytes = None
                else:
                    if "," in text_data:
                        text_data = text_data.split(",", 1)[1]
                    try:
                        frame_bytes = base64.b64decode(text_data)
                    except Exception:
                        frame_bytes = None

            if not frame_bytes:
                await websocket.send_json({
                    "success": False,
                    "error": "Empty or invalid frame payload",
                    "faces": [],
                })
                continue

            # Update threshold if requested
            if threshold_override is not None:
                video_proc.threshold = float(threshold_override)

            # Process frame
            t0 = time.perf_counter()
            try:
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None or frame.size == 0:
                    raise ValueError("Cannot decode image frame")

                result = video_proc.process_frame(frame)
                fps = round(1000.0 / max(result.inference_time_ms, 1.0), 1)

                # Format response
                faces_data = [f.model_dump() for f in result.faces]

                await websocket.send_json({
                    "success": True,
                    "face_count": result.face_count,
                    "faces": faces_data,
                    "inference_time_ms": result.inference_time_ms,
                    "fps": fps,
                    "image_width": int(frame.shape[1]),
                    "image_height": int(frame.shape[0]),
                    "timestamp": round(time.time(), 3),
                })
            except Exception as exc:
                await websocket.send_json({
                    "success": False,
                    "error": str(exc),
                    "faces": [],
                    "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                })

    except WebSocketDisconnect:
        # Client gracefully closed connection
        pass
    except Exception as exc:
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass


@router.post("/upload", response_model=VideoSummaryResponse)
async def upload_and_process_video(
    file: UploadFile = File(..., description="Video file (.mp4, .avi, .mov, .webm)"),
    stride: int = 2,
    frame_stride: Optional[int] = None,
    confidence_threshold: float = 0.5697,
    max_frames: int = 150,
):
    """
    Upload a video file for frame-by-frame face recognition analysis.
    Returns aggregated metrics, unique identities detected, and timeline events.
    """
    effective_stride = frame_stride if frame_stride is not None else stride
    valid_exts = [".mp4", ".avi", ".mov", ".webm", ".mkv"]
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in valid_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{ext}'. Allowed: {', '.join(valid_exts)}",
        )

    # Save to temporary file
    temp_dir = tempfile.gettempdir()
    temp_video_path = os.path.join(temp_dir, f"video_upload_{int(time.time())}{ext}")

    try:
        contents = await file.read()
        with open(temp_video_path, "wb") as f:
            f.write(contents)

        video_proc = VideoProcessor(
            detector=recognition_service.detector,
            aligner=recognition_service.aligner,
            embedder=recognition_service.embedder,
            classifier=recognition_service.classifier,
            threshold=float(confidence_threshold),
        )

        timeline: List[dict] = []
        unique_identities = set()
        latencies: List[float] = []
        total_faces = 0

        for res in video_proc.process_video_file(
            temp_video_path,
            frame_stride=effective_stride,
            max_frames=max_frames,
        ):
            latencies.append(res.inference_time_ms)
            total_faces += res.face_count

            # Collect identities
            for f in res.faces:
                if f.identity and f.identity.lower() != "unknown":
                    unique_identities.add(f.identity)

            # Sample timeline events (every 5th processed frame or when faces are present)
            if res.face_count > 0 or len(timeline) < 10:
                timeline.append({
                    "frame_index": res.frame_index,
                    "timestamp_ms": res.timestamp_ms,
                    "face_count": res.face_count,
                    "identities": [f.identity for f in res.faces],
                    "inference_time_ms": res.inference_time_ms,
                })

        avg_latency = round(sum(latencies) / max(len(latencies), 1), 2)
        avg_fps = round(1000.0 / max(avg_latency, 1.0), 1)

        return VideoSummaryResponse(
            success=True,
            filename=file.filename or "uploaded_video",
            total_frames_processed=len(latencies),
            duration_seconds=round(len(latencies) * (effective_stride / 30.0), 2),
            average_fps=avg_fps,
            average_latency_ms=avg_latency,
            unique_identities_detected=sorted(list(unique_identities)),
            timeline_sample=timeline[:30],
        )

    finally:
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass
