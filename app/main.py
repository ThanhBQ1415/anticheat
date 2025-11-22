from typing import Dict
import logging
import os
import asyncio
import time

import cv2  # noqa: F401  (ensure OpenCV is loaded early for some environments)

# Cấu hình logging để hiển thị INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Tắt access log của uvicorn (các log INFO về requests)
logging.getLogger("uvicorn.access").disabled = True
# Giảm log của VoiceDetector xuống WARNING để tránh spam console
logging.getLogger("app.monitoring.voice_detector").setLevel(logging.WARNING)
from fastapi import FastAPI, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, HTTPException, Form
import numpy as np
import cv2
from typing import Optional
import httpx

from .monitoring.face_detector import FaceDetector
from .monitoring.eye_tracker import EyeTracker
from .monitoring.voice_detector import VoiceDetector
from .session import SessionManager
from .auth import require_bearer_auth, require_session_id
from .models.schemas import ViolationType

app = FastAPI(title="Anti-cheat Service", version="1.0.0")

# CORS (adjust in later task if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global singletons (kept warm)
detectors: Dict[str, object] = {}
session_manager = SessionManager()

# Backend API URL
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080")

# Track last violation sent to avoid spam
_last_violation_sent: Dict[str, float] = {}
VIOLATION_COOLDOWN_SECONDS = 5.0  # Don't send same violation type more than once per 5 seconds


@app.on_event("startup")
def on_startup() -> None:
    # Initialize detectors once
    detectors["face"] = FaceDetector()
    detectors["eye"] = EyeTracker()
    detectors["voice"] = VoiceDetector()


@app.on_event("shutdown")
def on_shutdown() -> None:
    # Release detector resources
    face: FaceDetector = detectors.get("face")  # type: ignore
    if face:
        face.release()
    eye: EyeTracker = detectors.get("eye")  # type: ignore
    if eye:
        eye.release()
    voice: VoiceDetector = detectors.get("voice")  # type: ignore
    if voice:
        voice.release()
    session_manager.cleanup()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


async def send_violation_to_backend(exam_id: int, student_id: int, violation_type: ViolationType, message: str, session_id: str):
    """Send violation to backend API asynchronously"""
    try:
        violation_key = f"{exam_id}_{student_id}_{violation_type.value}"
        current_time = time.time()
        
        # Check cooldown
        if violation_key in _last_violation_sent:
            time_since_last = current_time - _last_violation_sent[violation_key]
            if time_since_last < VIOLATION_COOLDOWN_SECONDS:
                return  # Skip if within cooldown period
        
        url = f"{BACKEND_API_URL}/api/violation/log"
        payload = {
            "examId": exam_id,
            "studentId": student_id,
            "violationType": violation_type.value.upper(),
            "message": message,
            "sessionId": session_id
        }
        
        # Log the data being sent to backend
        logging.info(f"Sending violation to backend - URL: {url}, Payload: {payload}")
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                logging.info(f"Violation sent successfully to backend: {violation_type.value}, Response: {response.text}")
                _last_violation_sent[violation_key] = current_time
            else:
                logging.warning(f"Failed to send violation to backend: Status={response.status_code}, Response={response.text}, Payload={payload}")
    except Exception as e:
        logging.error(f"Error sending violation to backend: {e}")


@app.post("/api/anti-cheat/session/start")
async def start_session(
    exam_id: int = Body(...),
    student_id: int = Body(...),
    _=Depends(require_bearer_auth)
) -> Dict[str, str]:
    session_id = session_manager.create(exam_id=exam_id, student_id=student_id)
    return {"sessionId": session_id}


@app.post("/api/anti-cheat/session/stop")
def stop_session(sessionId: str, _=Depends(require_bearer_auth)) -> Dict[str, bool]:
    # simple body/query param; FE will send ?sessionId=... or JSON body via later enhancement
    ok = session_manager.stop(sessionId)
    return {"stopped": ok}

@app.post("/api/anti-cheat/frame")
async def analyze_frame(
    file: UploadFile = File(...),
    _=Depends(require_bearer_auth),
    x_session_id: str = Depends(require_session_id),
):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Unsupported image type")

    session_manager.touch(x_session_id)

    # Read image bytes
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    face: FaceDetector = detectors["face"]  # type: ignore
    eye: EyeTracker = detectors["eye"]  # type: ignore

    alerts = []
    metrics: Dict[str, float | bool] = {}
    face_present, face_data = face.detect_face(frame)

    if not face_present:
        alerts.append("no_face")
    else:
        metrics["faceConfidence"] = float(face_data["confidence"])

    looking_away, message = eye.is_looking_away(frame)
    if looking_away:
        alerts.append("looking_away")
        # Send violation to backend
        session_info = session_manager.get(x_session_id)
        if session_info and session_info.exam_id and session_info.student_id:
            asyncio.create_task(send_violation_to_backend(
                exam_id=session_info.exam_id,
                student_id=session_info.student_id,
                violation_type=ViolationType.EYE_GAZE,
                message=message or "Student looking away detected",
                session_id=x_session_id
            ))

    # Check for face presence violation
    if not face_present:
        session_info = session_manager.get(x_session_id)
        if session_info and session_info.exam_id and session_info.student_id:
            asyncio.create_task(send_violation_to_backend(
                exam_id=session_info.exam_id,
                student_id=session_info.student_id,
                violation_type=ViolationType.FACE_PRESENCE,
                message="Student face not detected",
                session_id=x_session_id
            ))

    response = {
        "alerts": alerts,
        "face": face_data if face_present else None,
        "metrics": metrics,
        "message": message if looking_away else None,
    }
    return response


@app.post("/api/anti-cheat/audio")
async def analyze_audio(
    file: UploadFile = File(...),
    _=Depends(require_bearer_auth),
    x_session_id: str = Depends(require_session_id),
):
    session_manager.touch(x_session_id)

    content_type = file.content_type or "application/octet-stream"
    if content_type not in ("application/octet-stream", "audio/raw", "audio/pcm"):
        raise HTTPException(status_code=400, detail="Unsupported audio type; expected raw PCM16 mono 16k")

    audio_bytes = await file.read()
    voice: VoiceDetector = detectors["voice"]  # type: ignore
    is_speech = voice.process_audio_frame(audio_bytes)

    alerts = []
    if is_speech:
        alerts.append("speech_detected")
        # Send violation to backend
        session_info = session_manager.get(x_session_id)
        if session_info and session_info.exam_id and session_info.student_id:
            asyncio.create_task(send_violation_to_backend(
                exam_id=session_info.exam_id,
                student_id=session_info.student_id,
                violation_type=ViolationType.VOICE,
                message="Human speech detected",
                session_id=x_session_id
            ))

    return {
        "alerts": alerts,
        "metrics": {"speech": is_speech},
    }
