import time
import uuid
from typing import Any


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self, exercise_id: str, reps: int, sets: int) -> str:
        sid = str(uuid.uuid4())
        self._sessions[sid] = {
            "id": sid,
            "exercise_id": exercise_id,
            "reps": reps,
            "sets": sets,
            "started_at": time.time(),
            "ended_at": None,
            "frame_idx": 0,
            "report": None,
        }
        return sid

    def get(self, sid: str) -> dict[str, Any] | None:
        return self._sessions.get(sid)

    def mark_ended(self, sid: str) -> float | None:
        sess = self._sessions.get(sid)
        if sess is None:
            return None
        ended = time.time()
        sess["ended_at"] = ended
        return ended

    def set_report(self, sid: str, report: dict[str, Any]) -> None:
        self._sessions[sid]["report"] = report

    def get_report(self, sid: str) -> dict[str, Any] | None:
        sess = self._sessions.get(sid)
        return sess["report"] if sess else None


store = SessionStore()
