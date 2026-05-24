"""
친근체 코칭 피드백 엔진.

멘토링 피드백:
- LLM 어투를 딱딱 → 친근/응원 톤으로 ("이 부분만 좀 더 고쳐봐요", "할 수 있어요")
- '안으로 굽었는데 밖으로 굽었으니 잘못되었다' 같은 부정형 표현 자제
- 오프라인 환경 대응 → API 실패 시 rule-based 친근체 메시지 자동 생성

두 가지 모드:
  1) online_gemini : Gemini 2.5 Flash + 친근체 system prompt
  2) offline_rule  : 템플릿 + 점수/오차 기반 친근체 메시지 (네트워크/키 불필요)
"""
from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional


FRIENDLY_SYSTEM = (
    "당신은 따뜻하고 응원하는 톤의 필라테스 코치입니다. "
    "사용자가 자신감을 가질 수 있도록 짧고 친근하게 말해주세요. "
    "반드시 한국어 존댓말로 답하고, 다음 규칙을 엄격히 지킵니다:\n"
    "1) '잘못', '틀렸다', '실패' 같은 부정적인 단어는 절대 쓰지 마세요. "
    "대신 '조금만 더', '한 끗만 다듬으면', '이미 잘하고 계세요' 같은 표현을 쓰세요.\n"
    "2) 정확히 3문장. 각 문장은 짧게 — 한 문장당 35자 이내. 전체 120자 이내.\n"
    "3) 1문장: 잘한 점 칭찬. 2문장: 개선할 핵심 한 가지만. 3문장: 다음에 시도할 짧은 큐.\n"
    "4) 숫자(각도, 점수)는 자연스럽게 한두 개만. 과하게 나열 금지.\n"
    "5) 모든 문장은 반드시 마침표 또는 물음표로 끝나야 합니다. 중간에 끊기면 안 됩니다."
)


def _short_summary_for_prompt(pose_kr: str, score_summary: Dict) -> str:
    """채점 결과를 LLM에게 줄 텍스트로 요약 (한국어)"""
    per_angle_acc = score_summary.get("per_angle_accuracy", {})
    per_angle_diff = score_summary.get("per_angle_mean_diff", {})
    angle_lines = []
    for key in per_angle_acc.keys():
        angle_lines.append(
            f"- {key}: 정답과 평균 {per_angle_diff.get(key, 0)}도 차이, "
            f"허용범위 통과율 {per_angle_acc.get(key, 0)}%"
        )
    return (
        f"자세: {pose_kr}\n"
        f"전체 점수: {score_summary.get('mean_score', 0)} / 100 ({score_summary.get('verdict','-')})\n"
        f"O/X 정량 정확도: {score_summary.get('ox_accuracy', 0)}%\n"
        f"가장 개선이 필요한 부분: {score_summary.get('top_issue_name_kr','-')}\n"
        f"각도별 통계:\n" + "\n".join(angle_lines)
    )


# -----------------------------------------------------------------------------
# Online — Gemini 2.5 Flash
# -----------------------------------------------------------------------------
def request_gemini_friendly(pose_kr: str, score_summary: Dict, api_key: str, timeout: int = 12) -> Optional[str]:
    user_block = _short_summary_for_prompt(pose_kr, score_summary)
    prompt = f"{FRIENDLY_SYSTEM}\n\n[채점 결과]\n{user_block}\n\n[당신의 친근한 코칭 3문장]"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    req = urllib.request.Request(
        api_url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, TimeoutError) as exc:
        print(f"[LLM] online 실패 → offline 폴백: {exc}")
        return None


# -----------------------------------------------------------------------------
# Offline — Rule-based 친근체 (네트워크/키 불필요)
# -----------------------------------------------------------------------------
_PRAISE_LINES = [
    "{pose} 동작 정말 잘 따라오셨어요!",
    "오늘 {pose} 자세 안정적으로 잡으셨네요!",
    "{pose} 흐름이 한결 부드러워졌어요!",
    "{pose}, 시작이 좋습니다!",
]
_PRAISE_HIGH = [
    "{pose} 완성도가 정말 높아요, 거의 정자세에 가까워요!",
    "{pose} 핵심 각도들이 정답에 딱 들어왔어요!",
    "{pose} 자세, 보고 있는 제가 다 시원해질 정도예요!",
]
_CUE_BY_ISSUE = {
    "knee": "다음에는 무릎 각도를 살짝 더 신경 써 보면 좋을 것 같아요.",
    "hip": "다음에는 고관절 굴곡/신전을 한 끗만 더 깊게 가져가 봐요.",
    "trunk": "다음에는 상체 정렬을 길게 늘인다는 느낌으로 가져가 보세요.",
}
_END_LINES = [
    "지금 페이스 그대로면 충분히 더 좋아질 수 있어요, 화이팅이에요!",
    "이미 거의 다 왔어요, 다음 세트도 같이 가봐요!",
    "한 번만 더 해보시면 분명 더 좋아질 거예요!",
]


def _verdict_to_praise(verdict: str, pose_kr: str) -> str:
    if verdict == "훌륭해요":
        return random.choice(_PRAISE_HIGH).format(pose=pose_kr)
    return random.choice(_PRAISE_LINES).format(pose=pose_kr)


def _build_cue(top_issue_key: Optional[str], top_issue_name_kr: str, mean_diff: float) -> str:
    base = _CUE_BY_ISSUE.get(top_issue_key or "", f"다음에는 {top_issue_name_kr}을(를) 한 끗만 더 다듬어 봐요.")
    if mean_diff > 25:
        base += " 평소보다 조금 더 크게 가동범위를 잡아 본다고 생각하시면 돼요."
    return base


def offline_friendly_feedback(pose_kr: str, score_summary: Dict) -> str:
    """네트워크 없이도 친근체 3문장을 생성"""
    mean_score = float(score_summary.get("mean_score", 0))
    ox_acc = float(score_summary.get("ox_accuracy", 0))
    verdict = score_summary.get("verdict", "좋아요")
    top_issue = score_summary.get("top_issue")
    top_issue_name = score_summary.get("top_issue_name_kr", "-")
    mean_diff = float(score_summary.get("per_angle_mean_diff", {}).get(top_issue, 0))

    s1 = _verdict_to_praise(verdict, pose_kr)
    if mean_score >= 85:
        s2 = f"이미 합격선({ox_acc:.0f}%)을 훌쩍 넘어서, 미세한 마무리만 다듬으면 완벽해질 거예요."
    elif mean_score >= 70:
        s2 = f"전반적으로 좋은데 {top_issue_name} 쪽이 살짝 아쉬워요, 그 부분만 가볍게 의식해 봐요."
    else:
        s2 = f"{top_issue_name}이(가) 정답에서 평균 {mean_diff:.0f}도 정도 차이가 나는 편이에요, 거기만 조금 더 신경 써 보면 좋아요."
    s3 = _build_cue(top_issue, top_issue_name, mean_diff) + " " + random.choice(_END_LINES)
    return f"{s1} {s2} {s3}"


# -----------------------------------------------------------------------------
# Unified entry point
# -----------------------------------------------------------------------------
def generate_feedback(
    pose_kr: str,
    score_summary: Dict,
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict:
    """
    Returns:
        {"text": str, "mode": "online_gemini" | "offline_rule"}
    """
    if prefer_online and api_key and api_key != "PASTE_YOUR_KEY_HERE":
        text = request_gemini_friendly(pose_kr, score_summary, api_key)
        if text:
            return {"text": text, "mode": "online_gemini"}
    text = offline_friendly_feedback(pose_kr, score_summary)
    return {"text": text, "mode": "offline_rule"}
