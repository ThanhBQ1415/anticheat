# Anti-Cheat Service

Python-based anti-cheat service for monitoring students during online exams.

## Features

- **Camera Monitoring**: Real-time face detection and eye gaze tracking
- **Voice Detection**: Detects human speech using VAD (Voice Activity Detection)
- **Face Presence Detection**: Monitors if student leaves camera view
- **Eye Gaze Detection**: Detects when student looks away for 5+ seconds
- **WebSocket Notifications**: Real-time violation notifications to frontend
- **REST API**: Endpoints for starting/stopping monitoring and checking violations

## Requirements

- Python 3.8+
- Camera access
- Microphone access
- OpenCV
- MediaPipe
- FastAPI
- WebSocket support

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install system dependencies for PyAudio:
   - **Linux**:
     ```bash
     sudo apt-get update
     sudo apt-get install portaudio19-dev python3-pyaudio
     ```
   - **macOS**:
     ```bash
     brew install portaudio
     ```
   - **Windows**: No additional system packages required.

3. Upgrade pip and install Python dependencies:
```bash
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

> If you install new dependencies later, make sure the virtual environment is activated before running `pip`.

## Running the Service

With the virtual environment activated (`source .venv/bin/activate`):
```bash
# Optional: protect APIs with a bearer token
export ANTICHEAT_BEARER_TOKEN="your-secret-token"

# Start FastAPI (HTTP server)
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8081
```

The service will run on `http://localhost:8081`

### HTTP Upload Flow (FE → BE)

1. FE gọi `POST /api/anti-cheat/session/start` để lấy `sessionId`.
2. FE định kỳ gửi:
   - Ảnh (JPEG) qua `POST /api/anti-cheat/frame` (multipart/form-data), header `X-Session-Id: <sessionId>`
   - Khối audio PCM16 mono 16k qua `POST /api/anti-cheat/audio`, header `X-Session-Id: <sessionId>`
3. Khi kết thúc, FE gọi `POST /api/anti-cheat/session/stop?sessionId=<sessionId>`.

## API Endpoints

### POST /api/anti-cheat/session/start
Tạo session theo dõi.

Request headers:
- Authorization: `Bearer <ANTICHEAT_BEARER_TOKEN>` (nếu biến môi trường được thiết lập)

Response:
```json
{
  "sessionId": "session_123"
}
```

Curl:
```bash
curl -X POST http://localhost:8081/api/anti-cheat/session/start \
  -H "Authorization: Bearer $ANTICHEAT_BEARER_TOKEN"
```

### POST /api/anti-cheat/session/stop
Dừng session theo dõi.

Query:
```
?sessionId=session_123
```

Request headers:
- Authorization: `Bearer <ANTICHEAT_BEARER_TOKEN>` (nếu bật)

Curl:
```bash
curl -X POST "http://localhost:8081/api/anti-cheat/session/stop?sessionId=session_123" \
  -H "Authorization: Bearer $ANTICHEAT_BEARER_TOKEN"
```

### POST /api/anti-cheat/frame
Gửi 1 khung hình (JPEG/PNG) để phân tích khuôn mặt và ánh nhìn.

Headers:
- Authorization: `Bearer <ANTICHEAT_BEARER_TOKEN>` (nếu bật)
- X-Session-Id: `<sessionId>`

Multipart form-data:
- file: ảnh `image/jpeg` hoặc `image/png`

Response:
```json
{
  "alerts": ["no_face", "looking_away"],
  "face": { "x": 100, "y": 80, "width": 120, "height": 120, "confidence": 0.85 },
  "metrics": { "faceConfidence": 0.85 },
  "message": "Looking away detected for 5.2 seconds"
}
```

Curl:
```bash
curl -X POST http://localhost:8081/api/anti-cheat/frame \
  -H "Authorization: Bearer $ANTICHEAT_BEARER_TOKEN" \
  -H "X-Session-Id: session_123" \
  -F "file=@/path/to/frame.jpg;type=image/jpeg"
```

### POST /api/anti-cheat/audio
Gửi 1 khối audio thô (raw PCM16 mono 16k) để phát hiện giọng nói.

Headers:
- Authorization: `Bearer <ANTICHEAT_BEARER_TOKEN>` (nếu bật)
- X-Session-Id: `<sessionId>`

Multipart form-data:
- file: `application/octet-stream` (bytes PCM16 mono 16k)

Response:
```json
{
  "alerts": ["speech_detected"],
  "metrics": { "speech": true }
}
```

Curl (ví dụ với file nhị phân đã có sẵn):
```bash
curl -X POST http://localhost:8081/api/anti-cheat/audio \
  -H "Authorization: Bearer $ANTICHEAT_BEARER_TOKEN" \
  -H "X-Session-Id: session_123" \
  -F "file=@/path/to/chunk.pcm;type=application/octet-stream"
```

### GET /health
Health check endpoint.

> Ghi chú: Bản này sử dụng HTTP upload thay vì WebSocket để truyền ảnh/âm thanh.

## Violation Types

- `eye_gaze`: Student looking away for 5+ seconds
- `voice`: Human speech detected
- `face_presence`: Student left camera view for 3+ seconds

## Configuration

The service can be configured by modifying the following constants in the code:

- Eye gaze threshold: `LOOK_AWAY_THRESHOLD` (degrees)
- Eye gaze duration: `LOOK_AWAY_DURATION` (seconds)
- Face absence threshold: `face_absence_threshold` (seconds)
- Voice detection sensitivity: VAD aggressiveness mode (0-3)

## Notes

- The service requires direct access to camera and microphone
- For web applications, camera/microphone access is handled by the browser
- The service should run on the same machine as the browser or be accessible via network
- Camera index can be configured (default: 0)

## Troubleshooting

1. **Camera not found**: Check if camera is connected and accessible
2. **Microphone not found**: Check if microphone is connected and permissions are granted
3. **WebSocket connection failed**: Check if service is running on port 8081
4. **Violations not detected**: Check camera/microphone permissions and lighting conditions

