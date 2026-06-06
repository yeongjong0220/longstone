import logging
import time

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import ai_bridge, ws

logger = logging.getLogger("uvicorn.error")
from app.schemas import (
    Exercise,
    SessionCreateReq,
    SessionCreateResp,
    SessionEndResp,
)
from app.sessions import store

app = FastAPI(title="Longstone Coach Backend (mock)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws.router)


_EXERCISES: list[Exercise] = [
    Exercise(id="the_seal", name="The Seal", reps_default=8, sets_default=2),
    Exercise(id="spine_stretch", name="Spine Stretch", reps_default=10, sets_default=3),
    Exercise(id="bridging", name="Bridging", reps_default=12, sets_default=3),
]


@app.get("/api/exercises", response_model=list[Exercise])
def list_exercises() -> list[Exercise]:
    return _EXERCISES


@app.post("/api/sessions", response_model=SessionCreateResp)
def create_session(req: SessionCreateReq) -> SessionCreateResp:
    if not any(ex.id == req.exercise_id for ex in _EXERCISES):
        raise HTTPException(status_code=404, detail="unknown exercise_id")
    sid = store.create(req.exercise_id, req.reps, req.sets)
    return SessionCreateResp(session_id=sid)


def _build_report(session_id: str) -> None:
    sess = store.get(session_id)
    if sess is None:
        return
    summary = {
        "exercise_id": sess["exercise_id"],
        "frame_count": sess["frame_idx"],
        "angle_history": sess.get("angle_history", []),     # v6 분포 채점에 필요
        "reps_target": sess.get("reps", 0),
        "sets_target": sess.get("sets", 0),
    }
    t0 = time.perf_counter()
    report = ai_bridge.generate_coaching(summary)
    # PLAN.md §검증 W11: LLM 호출 < 5초 목표
    logger.info("generate_coaching %s: %.2fs", session_id, time.perf_counter() - t0)
    store.set_report(session_id, report)


@app.post("/api/sessions/{session_id}/end", response_model=SessionEndResp)
def end_session(
    session_id: str, background_tasks: BackgroundTasks
) -> SessionEndResp:
    ended = store.mark_ended(session_id)
    if ended is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    background_tasks.add_task(_build_report, session_id)
    return SessionEndResp(session_id=session_id, ended_at=ended)


@app.get("/api/sessions/{session_id}/report")
def get_report(session_id: str) -> JSONResponse:
    if store.get(session_id) is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    report = store.get_report(session_id)
    if report is None:
        return JSONResponse(status_code=202, content={"status": "pending"})
    return JSONResponse(
        status_code=200, content={"status": "ready", "report": report}
    )
