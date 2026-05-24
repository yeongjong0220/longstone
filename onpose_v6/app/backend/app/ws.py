import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app import ai_bridge
from app.schemas import CoachingFrame, LandmarkFrame
from app.sessions import store

router = APIRouter()
logger = logging.getLogger("uvicorn.error")


@router.websocket("/ws/pose/{session_id}")
async def pose_ws(ws: WebSocket, session_id: str) -> None:
    await ws.accept()
    sess = store.get(session_id)
    if sess is None:
        await ws.close(code=4404, reason="unknown session")
        return

    try:
        while True:
            msg = await ws.receive_json()
            try:
                frame = LandmarkFrame.model_validate(msg)
            except ValidationError as e:
                logger.warning("invalid LandmarkFrame: %s", e.errors()[:1])
                continue

            sess["t"] = frame.t
            t0 = time.perf_counter()
            result = ai_bridge.analyze_frame(frame.landmarks, sess)
            analyze_ms = (time.perf_counter() - t0) * 1000

            # AI 팀 함수는 _dbg를 모름. 서버 wrap이 측정값을 주입.
            existing_dbg = result.get("_dbg") or result.get("dbg") or {}
            result["_dbg"] = {**existing_dbg, "analyze_ms": round(analyze_ms, 2)}

            try:
                coaching = CoachingFrame.model_validate(result)
            except ValidationError as e:
                logger.warning("invalid CoachingFrame: %s", e.errors()[:1])
                continue

            await ws.send_json(coaching.model_dump(by_alias=True, exclude_none=True))
    except WebSocketDisconnect:
        return
