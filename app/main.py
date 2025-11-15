from typing import Dict
import logging

import cv2  # noqa: F401  (ensure OpenCV is loaded early for some environments)

# Cấu hình logging để hiển thị INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Tắt access log của uvicorn (các log INFO về requests)
logging.getLogger("uvicorn.access").disabled = True
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, HTTPException, Form
import numpy as np
import cv2
from typing import Optional

from .monitoring.face_detector import FaceDetector
from .monitoring.eye_tracker import EyeTracker
# from .monitoring.voice_detector import VoiceDetector
from .session import SessionManager
from .auth import require_bearer_auth, require_session_id

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


@app.on_event("startup")
def on_startup() -> None:
    # Initialize detectors once
    detectors["face"] = FaceDetector()
    detectors["eye"] = EyeTracker()
    # detectors["voice"] = VoiceDetector()


@app.on_event("shutdown")
def on_shutdown() -> None:
    # Release detector resources
    face: FaceDetector = detectors.get("face")  # type: ignore
    if face:
        face.release()
    eye: EyeTracker = detectors.get("eye")  # type: ignore
    if eye:
        eye.release()
    # voice: VoiceDetector = detectors.get("voice")  # type: ignore
    # if voice:
    #     voice.release()
    session_manager.cleanup()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/anti-cheat/session/start")
def start_session(_=Depends(require_bearer_auth)) -> Dict[str, str]:
    session_id = session_manager.create()
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

    response = {
        "alerts": alerts,
        "face": face_data if face_present else None,
        "face": None,
        "metrics": metrics,
        "message": None,
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
    # voice: VoiceDetector = detectors["voice"]  # type: ignore
    # is_speech = voice.process_audio_frame(audio_bytes)

    alerts = []
    # if is_speech:
    #     alerts.append("speech_detected")

    return {
        "alerts": alerts,
        "metrics": {"speech": False},  # {"speech": bool(is_speech)},
    }
