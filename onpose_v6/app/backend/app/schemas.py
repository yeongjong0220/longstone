from typing import Literal

from pydantic import BaseModel, Field


class Exercise(BaseModel):
    id: str
    name: str
    reps_default: int
    sets_default: int


class SessionCreateReq(BaseModel):
    exercise_id: str
    reps: int = Field(gt=0)
    sets: int = Field(gt=0)


class SessionCreateResp(BaseModel):
    session_id: str


class LandmarkFrame(BaseModel):
    t: float
    landmarks: list[list[float]]


class FeedbackItem(BaseModel):
    level: Literal["ok", "warn", "err"]
    msg: str


class CoachingFrame(BaseModel):
    t: float
    phase: str
    rep_count: int
    set_count: int
    angles: dict[str, float]
    score: int
    status: Literal["good", "warn", "err"]
    feedback: list[FeedbackItem]
    # 서버에서 주입하는 측정값. AI 팀 함수는 무지(無知), ws.py가 wrap 후 채움.
    dbg: dict[str, float] | None = Field(default=None, alias="_dbg")

    model_config = {"populate_by_name": True}


class Report(BaseModel):
    score_avg: int
    good_points: list[str]
    improvements: list[str]
    llm_msg: str


class SessionEndResp(BaseModel):
    session_id: str
    ended_at: float
