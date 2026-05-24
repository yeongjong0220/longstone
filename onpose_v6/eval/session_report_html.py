"""
세션 결과를 한 페이지 HTML 리포트로 만들어 PPT / 공유에 그대로 사용.

reports/session_*.json + session_history.json 을 읽어
reports/report_*.html 로 생성.

브라우저에서 열면:
- 자세, 점수, O/X, 친근체 코칭
- 각도별 막대 (가중치 적용)
- 자세별 트렌드 그래프 (히스토리)
- Latency 표
모두 표시.

사용:
  python eval/session_report_html.py                 # 가장 최근 세션
  python eval/session_report_html.py --all           # 모든 세션
  python eval/session_report_html.py --session reports/session_20260523_xxx.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def _color_for(score: float) -> str:
    if score >= 85: return "#5ce0a3"
    if score >= 70: return "#5ed4f0"
    if score >= 55: return "#f5c34a"
    return "#f06b6b"


def _verdict_class(score: float) -> str:
    if score >= 85: return "good"
    if score >= 70: return "ok"
    if score >= 55: return "warn"
    return "bad"


def render_session(session_json: Dict, history: List[Dict]) -> str:
    pose_kr = session_json.get("score", {}).get("verdict_pose_kr", "")
    rubric_name = session_json.get("pose", "Unknown")
    score = session_json.get("score", {})
    fb = session_json.get("feedback", {})
    latency = session_json.get("latency", {}).get("summary", {})
    ts = session_json.get("timestamp", "")
    mean_score = score.get("mean_score", 0)
    ox = score.get("ox_accuracy", 0)
    verdict = score.get("verdict", "-")
    per_acc = score.get("per_angle_accuracy", {})
    per_diff = score.get("per_angle_mean_diff", {})
    pose_kr_map = {"The_Seal": "더 씰", "Bridging": "브릿징", "Spine_Stretch": "스파인 스트레치"}
    pose_kr = pose_kr_map.get(rubric_name, rubric_name)

    # 자세별 history 정리
    by_pose: Dict[str, List[float]] = {}
    for h in history:
        kr = h.get("pose_kr", "?")
        by_pose.setdefault(kr, []).append(float(h.get("score", 0)))
    trend_rows = "".join(
        f"<li><b>{kr}</b>: {len(scs)}회, 평균 <span style='color:{_color_for(sum(scs)/len(scs))}'>{sum(scs)/len(scs):.1f}점</span></li>"
        for kr, scs in by_pose.items()
    ) or "<li>아직 기록이 없어요</li>"

    # 각도 막대 SVG (간단한 막대 차트)
    bars = ""
    for key, pct in per_acc.items():
        diff = per_diff.get(key, 0)
        color = _color_for(pct)
        bars += f"""
        <div class="bar-row">
          <div class="bar-label">{key}</div>
          <div class="bar-bg"><div class="bar-fill" style="width:{min(100, max(0, pct))}%; background:{color}"></div></div>
          <div class="bar-val">{pct:.0f}%  <span class="diff">Δ{diff:.1f}°</span></div>
        </div>"""

    # Latency 표
    lat_rows = ""
    for name, stats in latency.items():
        if "mean_ms" in stats:
            lat_rows += f"<tr><td>{name}</td><td>{stats.get('n', 0)}</td><td>{stats.get('mean_ms', 0):.2f}</td><td>{stats.get('p95_ms', 0):.2f}</td><td>{stats.get('max_ms', 0):.2f}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="UTF-8">
<title>OnPose v6 — 세션 리포트 ({rubric_name})</title>
<style>
  body {{ font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; background: #1a1a26; color: #f0f0f0; max-width: 980px; margin: 0 auto; padding: 30px 40px; }}
  h1 {{ color: #7ec0ff; margin: 0 0 4px 0; }}
  h2 {{ color: #ffd66b; border-bottom: 2px solid #353550; padding-bottom: 6px; margin-top: 36px; }}
  .meta {{ color: #999; font-size: 14px; margin-bottom: 24px; }}
  .score-card {{ background: linear-gradient(135deg, #2a2a3e, #1f1f30); border-radius: 14px; padding: 28px 32px; display: flex; gap: 36px; align-items: center; }}
  .score-big {{ font-size: 80px; font-weight: 700; color: {_color_for(mean_score)}; line-height: 1; }}
  .score-meta div {{ margin: 6px 0; font-size: 17px; }}
  .verdict.good {{ color: #5ce0a3; }} .verdict.ok {{ color: #5ed4f0; }}
  .verdict.warn {{ color: #f5c34a; }} .verdict.bad {{ color: #f06b6b; }}
  .ox {{ color: {_color_for(ox)}; font-weight: 600; }}
  .feedback {{ background: #232336; border-left: 4px solid #ffd66b; padding: 18px 22px; border-radius: 6px; margin-top: 24px; line-height: 1.6; }}
  .bar-row {{ display: flex; align-items: center; margin: 10px 0; gap: 10px; }}
  .bar-label {{ width: 110px; font-weight: 600; }}
  .bar-bg {{ flex: 1; background: #353550; height: 24px; border-radius: 4px; overflow: hidden; }}
  .bar-fill {{ height: 100%; transition: width 0.5s; }}
  .bar-val {{ width: 120px; text-align: right; }}
  .diff {{ color: #888; font-size: 13px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #353550; }}
  th {{ background: #2a2a3e; color: #ffd66b; }}
  ul {{ line-height: 1.8; }}
  .footer {{ text-align: center; color: #666; margin-top: 50px; padding-top: 20px; border-top: 1px solid #2a2a3e; font-size: 13px; }}
</style>
</head><body>
<h1>OnPose v6 — 세션 리포트</h1>
<div class="meta">{ts} &nbsp;|&nbsp; 자세: <b>{pose_kr}</b> ({rubric_name}) &nbsp;|&nbsp; 모드: {fb.get('mode', '-')}</div>

<div class="score-card">
  <div class="score-big">{mean_score:.0f}</div>
  <div class="score-meta">
    <div>가중 점수: <b style="color:{_color_for(mean_score)}">{mean_score:.1f} / 100</b></div>
    <div>O/X 정확도: <b class="ox">{ox:.1f}%</b></div>
    <div>평가: <b class="verdict {_verdict_class(mean_score)}">{verdict}</b></div>
    <div>분석 프레임: {score.get('n_frames', 0)}</div>
    <div>가장 개선이 필요한 부분: <b>{score.get('top_issue_name_kr', '-')}</b></div>
  </div>
</div>

<div class="feedback">
  💬 <b>AI 코치 한 마디</b><br>{fb.get('text', '')}
</div>

<h2>각도별 정확도 (가중치 적용)</h2>
{bars}

<h2>자세별 세션 트렌드 (전체 히스토리)</h2>
<ul>{trend_rows}</ul>

<h2>컴포넌트 별 Latency</h2>
<table>
  <tr><th>Component</th><th>n</th><th>mean(ms)</th><th>p95(ms)</th><th>max(ms)</th></tr>
  {lat_rows}
</table>

<div class="footer">OnPose v6 — On-device pose coaching with friendly feedback &copy; 2026</div>
</body></html>"""
    return html


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, default=None,
                    help="특정 session JSON 파일 (없으면 가장 최근)")
    ap.add_argument("--all", action="store_true", help="모든 session에 대해 생성")
    ap.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    args = ap.parse_args()

    if args.all:
        targets = sorted(REPORTS_DIR.glob("session_*.json"))
    elif args.session:
        targets = [args.session]
    else:
        candidates = sorted(REPORTS_DIR.glob("session_*.json"))
        if not candidates:
            print("[error] no session JSON found in reports/")
            return 1
        targets = [candidates[-1]]

    history_path = REPORTS_DIR / "session_history.json"
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = []

    for src in targets:
        try:
            session = json.loads(src.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[skip] {src.name}: {exc}")
            continue
        html = render_session(session, history)
        out_path = args.out_dir / f"report_{src.stem}.html"
        out_path.write_text(html, encoding="utf-8")
        print(f"[ok] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
