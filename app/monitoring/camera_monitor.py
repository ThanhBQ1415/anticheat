import cv2
import threading
import time
import queue
import base64
from typing import Optional
import numpy as np
from .face_detector import FaceDetector
from .eye_tracker import EyeTracker
from .voice_detector import VoiceDetector


class CameraMonitor:
    def __init__(self, violation_queue: Optional[queue.Queue] = None):
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Detectors
        self.face_detector = FaceDetector()
        self.eye_tracker = EyeTracker()
        self.voice_detector = VoiceDetector()
        
        # Violation queue for async communication
        self.violation_queue = violation_queue or queue.Queue()
        
        # State tracking
        self.last_face_detection_time: Optional[float] = None
        self.face_absence_threshold = 3.0  # seconds
        self.frame_rate = 30
        self.frame_interval = 1.0 / self.frame_rate
        self.last_voice_check_time = 0.0
        self.voice_check_interval = 1.0  # Check voice every second
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.monitor_start_time: float = 0.0
        self.startup_grace_period = 5.0  # seconds to ignore violations after start
        self.frame_queue = queue.Queue(maxsize=10)  # Queue for frames from frontend

    def start_monitoring(self):
        """Start monitoring with frames from frontend."""
        if self.is_monitoring:
            print("[CameraMonitor] Monitoring already started")
            return

        try:
            print("[CameraMonitor] Starting monitoring...")
            # Start voice monitoring (may fail if microphone is not available)
            try:
                self.voice_detector.start_monitoring()
                print("[CameraMonitor] Voice monitoring started")
            except Exception as mic_error:
                print(f"[CameraMonitor] Warning: Could not start voice monitoring: {mic_error}")
                # Continue without voice detection
            
            # Start frame processing thread (waits for frames from frontend)
            self.is_monitoring = True
            self.monitor_start_time = time.time()
            self.last_voice_check_time = self.monitor_start_time
            self.last_face_detection_time = self.monitor_start_time
            self.monitoring_thread = threading.Thread(target=self._process_frames, daemon=True)
            self.monitoring_thread.start()
            print("[CameraMonitor] Monitoring started successfully, waiting for frames from frontend...")
            
        except Exception as e:
            print(f"[CameraMonitor] Error starting monitoring: {e}")
            self.is_monitoring = False

    def _process_frames(self):
        """Process frames from frontend in background thread."""
        frame_count = 0
        print("[CameraMonitor] Frame processing thread started")
        while self.is_monitoring:
            try:
                # Wait for frame from frontend (with timeout to check if still monitoring)
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    # Log every 10 seconds if no frames received
                    if frame_count == 0:
                        print("[CameraMonitor] Waiting for frames from frontend...")
                    continue
                
                if frame is None:
                    continue
                
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames (~1 second at 30fps)
                    print(f"[CameraMonitor] Processing frame #{frame_count}")
                
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Process frame for violations
                self._process_frame(frame)
                
            except Exception as e:
                print(f"[CameraMonitor] Error processing frames: {e}")
                import traceback
                traceback.print_exc()
                break
        print("[CameraMonitor] Frame processing thread stopped")
    
    def receive_frame(self, frame_data: str):
        """Receive frame from frontend (base64 encoded image)."""
        try:
            # Decode base64 image
            if frame_data.startswith('data:image'):
                # Remove data URL prefix if present
                frame_data = frame_data.split(',')[1]
            
            image_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Put frame in queue (non-blocking, drop if queue is full)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    print("[CameraMonitor] Warning: Frame queue is full, dropping frame")
            else:
                print("[CameraMonitor] Warning: Failed to decode frame from base64")
        except Exception as e:
            print(f"[CameraMonitor] Error receiving frame: {e}")
            import traceback
            traceback.print_exc()

    def _process_frame(self, frame: np.ndarray):
        """Process frame to detect violations."""
        current_time = time.time()

        # Skip violation checks during initial grace period to allow user setup.
        if current_time - self.monitor_start_time < self.startup_grace_period:
            self.last_face_detection_time = current_time
            self.last_voice_check_time = current_time
            return
        
        # Log first frame after grace period
        if not hasattr(self, '_first_frame_logged'):
            print(f"[CameraMonitor] Processing first frame after grace period (frame shape: {frame.shape})")
            self._first_frame_logged = True
        
        # 1. Face presence detection
        is_face_present, face_data = self.face_detector.detect_face(frame)
        
        if not is_face_present:
            if self.last_face_detection_time is None:
                self.last_face_detection_time = current_time
            else:
                # Check if face has been absent for threshold duration
                absence_duration = current_time - self.last_face_detection_time
                if absence_duration >= self.face_absence_threshold:
                    # Put violation in queue
                    try:
                        self.violation_queue.put_nowait((
                            "face_presence",
                            f"Face not detected for {absence_duration:.1f} seconds"
                        ))
                    except queue.Full:
                        print("Violation queue is full")
                    return
        else:
            self.last_face_detection_time = current_time
            
            # 2. Eye gaze detection
            is_looking_away, message = self.eye_tracker.is_looking_away(frame)
            if is_looking_away:
                # Put violation in queue
                try:
                    self.violation_queue.put_nowait(("eye_gaze", message))
                except queue.Full:
                    print("Violation queue is full")
                return
        
        # 3. Voice detection (check periodically, not every frame)
        if current_time - self.last_voice_check_time >= self.voice_check_interval:
            self.last_voice_check_time = current_time
            if self.voice_detector.is_human_speech_detected():
                # Put violation in queue
                try:
                    self.violation_queue.put_nowait(("voice", "Human speech detected"))
                except queue.Full:
                    print("Violation queue is full")
                return

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame (from latest received frame)."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop_monitoring(self):
        """Stop monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        self.voice_detector.stop_monitoring()
        self.last_face_detection_time = None
        with self.frame_lock:
            self.latest_frame = None
        self.monitor_start_time = 0.0
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def release(self):
        """Release all resources."""
        self.stop_monitoring()
        self.face_detector.release()
        self.eye_tracker.release()
        self.voice_detector.release()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest captured frame."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def is_camera_open(self) -> bool:
        """Check if monitoring is active (frames are being received)."""
        return self.is_monitoring

