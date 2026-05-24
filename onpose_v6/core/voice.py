"""
TTS / 사운드 효과 모듈.

3가지 모드:
  - "sound"  : 짧은 비프/차임벨 (기본, 가장 자연스러움)
  - "tts"    : Windows SAPI / edge-tts (한국어 음성)
  - "off"    : 무음

CLI에서 --voice  → tts 모드
        --sound  → sound 모드 (기본)
        무옵션   → off
"""
from __future__ import annotations

import platform
import subprocess
import threading
from queue import Queue
from typing import Optional


def _try_winsound():
    try:
        import winsound
        return winsound
    except Exception:
        return None


class VoiceCoach:
    def __init__(self, mode: str = "off", lang: str = "ko-KR",
                 prefer_edge: bool = False) -> None:
        self.mode = mode             # "sound" | "tts" | "off"
        self.lang = lang
        self.prefer_edge = prefer_edge
        self._q: Queue = Queue()
        self._thread = None
        self._winsound = _try_winsound()
        self._has_edge = self._check_edge()
        if mode in ("sound", "tts"):
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    @staticmethod
    def _check_edge() -> bool:
        try:
            import edge_tts   # noqa
            return True
        except ImportError:
            return False

    # 외부 인터페이스
    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def say(self, text: str, sound: str = "info") -> None:
        """텍스트 또는 사운드 큐에 추가.
        mode='sound' → sound 키워드로 효과음 재생
        mode='tts'   → text를 음성으로 읽음
        """
        if self.mode == "off":
            return
        self._q.put((text, sound))

    # 컨벤션 메서드 (메인 코드에서 부르기 좋게)
    def say_intro(self, pose_kr: str) -> None:
        self.say(f"{pose_kr} 자세를 시작합니다. 자세를 취해 주세요.", sound="start")

    def say_start_capture(self) -> None:
        self.say("측정을 시작합니다", sound="ding")

    def say_finish(self) -> None:
        self.say("측정이 끝났어요. 분석 중입니다.", sound="done")

    def say_score(self, score: float, verdict: str) -> None:
        sound = "win" if score >= 85 else "ok" if score >= 70 else "info"
        if score >= 85:
            self.say(f"훌륭해요! {int(score)}점입니다.", sound=sound)
        elif score >= 70:
            self.say(f"좋아요. {int(score)}점이에요.", sound=sound)
        else:
            self.say(f"조금만 더 해봐요. {int(score)}점이에요.", sound=sound)

    def stop(self) -> None:
        self.mode = "off"

    # 내부 워커
    def _worker(self) -> None:
        while True:
            text, sound = self._q.get()
            if self.mode == "off":
                continue
            try:
                if self.mode == "sound":
                    self._play_sound(sound)
                elif self.mode == "tts":
                    if self.prefer_edge and self._has_edge:
                        self._speak_edge(text)
                    else:
                        self._speak_sapi(text)
            except Exception as e:
                print(f"[voice] failed: {e}")

    # ===== 사운드 효과 (가장 자연스러움) =====
    def _play_sound(self, kind: str) -> None:
        """짧은 톤 시퀀스로 청각 신호. 외부 의존성 0."""
        if self._winsound is None:
            return
        wave = {
            "ding":  [(880, 80)],                          # 측정 시작
            "start": [(660, 80), (880, 120)],              # 자세 안내
            "done":  [(700, 80), (550, 80), (700, 80)],    # 측정 끝
            "win":   [(660, 80), (880, 80), (1100, 160)],  # 훌륭한 점수 (상승)
            "ok":    [(660, 80), (880, 120)],              # 좋은 점수
            "info":  [(550, 80)],                          # 일반 안내
            "warn":  [(440, 100), (350, 120)],             # 주의
        }.get(kind, [(550, 80)])
        for freq, ms in wave:
            try:
                self._winsound.Beep(int(freq), int(ms))
            except Exception:
                pass

    # ===== Windows SAPI =====
    def _speak_sapi(self, text: str) -> None:
        text = text.replace('"', "'").replace("\n", " ")
        if platform.system() == "Windows":
            # Heami 보다 살짝 빠른 속도 + 강조 톤
            ps = (
                f"Add-Type -AssemblyName System.Speech; "
                f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = 1; "                            # -10~10 (0보다 살짝 빠르게)
                f"$s.SelectVoiceByHints('Female', 0, 0, [System.Globalization.CultureInfo]::new('ko-KR')); "
                f"$s.Speak(\"{text}\")"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True, timeout=15,
            )
        elif platform.system() == "Darwin":
            subprocess.run(["say", "-v", "Yuna", text], capture_output=True, timeout=15)
        else:
            subprocess.run(["espeak", "-v", "ko", text], capture_output=True, timeout=15)

    # ===== Edge-TTS (인터넷 필요, 매우 자연스러움) =====
    def _speak_edge(self, text: str) -> None:
        """Microsoft Edge 신경망 TTS — 인터넷 필요."""
        try:
            import asyncio
            import edge_tts
            import tempfile, os

            async def synth():
                voice = "ko-KR-SunHiNeural"     # 자연스러운 여성
                communicate = edge_tts.Communicate(text, voice, rate="+10%")
                fd, path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
                await communicate.save(path)
                return path

            path = asyncio.run(synth())
            # Windows 기본 재생 (PowerShell System.Media.SoundPlayer는 mp3 미지원이라 MediaPlayer 사용)
            if platform.system() == "Windows":
                ps_lines = [
                    "Add-Type -AssemblyName presentationCore",
                    "$p = New-Object System.Windows.Media.MediaPlayer",
                    f"$p.Open([System.Uri]::new('{path}'))",
                    "$p.Play()",
                    "Start-Sleep -Milliseconds 200",
                    "while ($p.NaturalDuration.HasTimeSpan -eq $false) {Start-Sleep -Milliseconds 50}",
                    "Start-Sleep -Seconds $p.NaturalDuration.TimeSpan.TotalSeconds",
                ]
                ps = "; ".join(ps_lines)
                subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                               capture_output=True, timeout=20)
            try:
                os.unlink(path)
            except Exception:
                pass
        except Exception as exc:
            print(f"[edge-tts] failed: {exc}, fallback to sound")
            self._play_sound("info")
