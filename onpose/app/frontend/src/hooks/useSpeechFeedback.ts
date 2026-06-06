import { useEffect, useRef } from "react"

const REPEAT_INTERVAL_MS = 10000
const GAP_AFTER_SPEAK_MS = 2000
const POLL_MS = 300
const KEEPALIVE_MS = 5000
const VOICE_LOAD_RETRY_MS = 500
const VOICE_LOAD_MAX_RETRIES = 12

function pickPreferredVoice(): SpeechSynthesisVoice | null {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return null
  const voices = window.speechSynthesis.getVoices()
  if (voices.length === 0) return null
  const ko = voices.filter((v) => v.lang.toLowerCase().startsWith("ko"))
  if (ko.length === 0) {
    console.warn("[TTS] no Korean voice. Available langs:", Array.from(new Set(voices.map((v) => v.lang))))
    return null
  }
  // 1순위: Google 한국의 (ko-KR)
  const googleKoKR = ko.find(
    (v) => v.name.toLowerCase().includes("google") && v.lang.toLowerCase() === "ko-kr",
  )
  if (googleKoKR) {
    console.info("[TTS] picked Google ko-KR:", googleKoKR.name, googleKoKR.lang)
    return googleKoKR
  }
  // 2순위: Google 한국어 (lang 표기 다른 케이스)
  const googleAny = ko.find((v) => v.name.toLowerCase().includes("google"))
  if (googleAny) {
    console.info("[TTS] picked Google Korean:", googleAny.name, googleAny.lang)
    return googleAny
  }
  // 3순위: Heami 가 아닌 Korean voice → 마지막 fallback 으로 첫 번째
  const nonHeami = ko.find((v) => !v.name.toLowerCase().includes("heami"))
  const picked = nonHeami ?? ko[0]
  console.info("[TTS] picked fallback Korean voice:", picked.name, picked.lang)
  return picked
}

function makeUtterance(
  text: string,
  voice: SpeechSynthesisVoice | null,
  onComplete?: () => void,
): SpeechSynthesisUtterance {
  const u = new SpeechSynthesisUtterance(text)
  if (voice) {
    u.voice = voice
    u.lang = voice.lang
  } else {
    u.lang = "ko-KR"
  }
  u.rate = 1.05
  u.pitch = 1.0
  u.onstart = () => console.debug("[TTS] speaking:", text)
  u.onend = () => {
    console.debug("[TTS] ended:", text)
    onComplete?.()
  }
  u.onerror = (e: SpeechSynthesisErrorEvent) => {
    console.error("[TTS] utterance error:", e.error, "text:", text)
    onComplete?.()
  }
  return u
}

/**
 * 무음 워밍업 — speechSynthesis 엔진 초기화 트리거.
 * Home 카드 탭, Countdown 마운트, useSpeechFeedback 마운트에서 모두 호출.
 */
export function primeSpeech() {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return
  try {
    const warm = new SpeechSynthesisUtterance("준비")
    warm.volume = 0
    warm.rate = 1.5
    warm.lang = "ko-KR"
    window.speechSynthesis.speak(warm)
  } catch (e) {
    console.warn("[TTS] primeSpeech failed", e)
  }
}

/**
 * Web Speech API 실시간 피드백.
 * - 1순위 voice: Google 한국의 ko-KR (없으면 Google Korean → non-Heami → 첫번째)
 * - voiceschanged 이벤트 + 0.5초 폴링 (최대 6초) 으로 voice 로드 보장
 * - 발화 중에는 새 speak 호출을 skip — 문장 cutoff 방지, 끝까지 듣기 보장
 * - 마지막 utterance onend 후 GAP_AFTER_SPEAK_MS(2초) 경과 시에만 다음 발화
 * - 동일 메시지는 REPEAT_INTERVAL_MS(10초) 마다 자동 재발화
 * - 5초마다 pause/resume keep-alive — Chrome ~15초 cutoff 버그 회피
 * - utterance onstart/onend/onerror 로 모든 실패가 콘솔에 남음
 */
export function useSpeechFeedback(messages: string[], enabled: boolean = true) {
  const messagesRef = useRef<string[]>(messages)
  messagesRef.current = messages
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null)
  const lastSigRef = useRef<string>("")
  const lastSpokeEndAtRef = useRef<number>(0)

  // 음성 카탈로그 로드 — voiceschanged 이벤트 + 폴백 폴링 (Chrome on Windows 보강)
  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      console.warn("[TTS] speechSynthesis not available")
      return
    }
    const trySet = () => {
      const picked = pickPreferredVoice()
      if (picked) voiceRef.current = picked
      return Boolean(picked)
    }
    if (trySet()) return

    let retryCount = 0
    const retryId = window.setInterval(() => {
      retryCount++
      if (trySet() || retryCount >= VOICE_LOAD_MAX_RETRIES) {
        window.clearInterval(retryId)
        if (!voiceRef.current) {
          console.warn("[TTS] voice load gave up after", retryCount, "retries")
        }
      }
    }, VOICE_LOAD_RETRY_MS)

    const handler = () => {
      if (trySet()) {
        window.speechSynthesis.removeEventListener("voiceschanged", handler)
        window.clearInterval(retryId)
      }
    }
    window.speechSynthesis.addEventListener("voiceschanged", handler)
    return () => {
      window.speechSynthesis.removeEventListener("voiceschanged", handler)
      window.clearInterval(retryId)
    }
  }, [])

  useEffect(() => {
    if (!enabled) return
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return

    console.info("[TTS] hook enabled — primeSpeech + start poll")
    primeSpeech()
    lastSigRef.current = ""
    lastSpokeEndAtRef.current = 0

    let cancelled = false

    const markSpeakEnd = () => {
      lastSpokeEndAtRef.current = Date.now()
    }

    const speak = () => {
      const msgs = messagesRef.current
      if (!msgs?.length) return
      const cleaned = msgs.filter((m): m is string => Boolean(m))
      if (cleaned.length === 0) return

      // 발화 중이거나 큐가 차 있으면 skip — 끝까지 말하게 둠
      if (window.speechSynthesis.speaking || window.speechSynthesis.pending) return

      if (!voiceRef.current) voiceRef.current = pickPreferredVoice()

      const sig = cleaned.join("|")
      lastSigRef.current = sig

      console.debug("[TTS] speak tick", {
        count: cleaned.length,
        paused: window.speechSynthesis.paused,
        voice: voiceRef.current?.name ?? "<none>",
      })

      if (window.speechSynthesis.paused) {
        window.speechSynthesis.resume()
      }
      for (const msg of cleaned) {
        window.speechSynthesis.speak(makeUtterance(msg, voiceRef.current, markSpeakEnd))
      }
    }

    const poll = () => {
      if (cancelled) return
      const msgs = messagesRef.current
      if (!msgs?.some(Boolean)) return

      // 발화/큐 진행 중이면 끝까지 기다림
      if (window.speechSynthesis.speaking || window.speechSynthesis.pending) return

      const sig = msgs.filter(Boolean).join("|")
      const gap = Date.now() - lastSpokeEndAtRef.current
      const sigChanged = sig !== lastSigRef.current

      if (sigChanged && gap >= GAP_AFTER_SPEAK_MS) {
        speak()
      } else if (!sigChanged && gap >= REPEAT_INTERVAL_MS) {
        speak()
      }
    }

    const pollId = window.setInterval(poll, POLL_MS)

    // Chrome ~15초 cutoff 버그 keep-alive: 발화 중일 때 5초마다 pause/resume
    const keepAliveId = window.setInterval(() => {
      if (cancelled) return
      const ss = window.speechSynthesis
      if (ss.speaking && !ss.paused) {
        ss.pause()
        ss.resume()
      } else if (ss.paused) {
        ss.resume()
      }
    }, KEEPALIVE_MS)

    return () => {
      cancelled = true
      window.clearInterval(pollId)
      window.clearInterval(keepAliveId)
      if (window.speechSynthesis.speaking || window.speechSynthesis.pending) {
        window.speechSynthesis.cancel()
      }
      console.info("[TTS] hook disabled — cleanup")
    }
  }, [enabled])
}
