import secrets
import threading
import time
from typing import Dict, Optional


class SessionInfo:
    def __init__(self, session_id: str, created_at: float, exam_id: Optional[int] = None, student_id: Optional[int] = None):
        self.session_id = session_id
        self.created_at = created_at
        self.last_seen = created_at
        self.is_active = True
        self.exam_id = exam_id
        self.student_id = student_id


class SessionManager:
    def __init__(self, ttl_seconds: int = 60 * 60 * 4):
        self._sessions: Dict[str, SessionInfo] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds

    def create(self, exam_id: Optional[int] = None, student_id: Optional[int] = None) -> str:
        session_id = secrets.token_urlsafe(16)
        now = time.time()
        with self._lock:
            self._sessions[session_id] = SessionInfo(session_id, now, exam_id, student_id)
        return session_id

    def touch(self, session_id: str) -> Optional[SessionInfo]:
        with self._lock:
            info = self._sessions.get(session_id)
            if info and info.is_active:
                info.last_seen = time.time()
                return info
        return None

    def stop(self, session_id: str) -> bool:
        with self._lock:
            info = self._sessions.get(session_id)
            if not info:
                return False
            info.is_active = False
            return True

    def get(self, session_id: str) -> Optional[SessionInfo]:
        with self._lock:
            return self._sessions.get(session_id)

    def cleanup(self) -> int:
        """Remove expired sessions, returns count cleaned."""
        now = time.time()
        removed = 0
        with self._lock:
            for sid in list(self._sessions.keys()):
                info = self._sessions[sid]
                if (now - info.last_seen) > self._ttl_seconds or not info.is_active:
                    del self._sessions[sid]
                    removed += 1
        return removed


