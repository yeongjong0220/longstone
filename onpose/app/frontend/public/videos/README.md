# 전문가 영상

홈 화면에서 운동 카드를 선택하면 풀스크린으로 1회 재생되는 데모 영상.
PLAN.md §3 ③ "3대 UI 요소" 중 하나.

## 파일 규칙
백엔드 `/api/exercises`의 `id`와 1:1 매칭되는 파일명 사용. 확장자 `.mp4` 고정.

현재 운동 목록(`ui_ux/backend/app/main.py:_EXERCISES`):
- `the_seal.mp4`
- `spine_stretch.mp4`
- `bridging.mp4`

## 동작
- `<video src="/videos/{id}.mp4" autoplay playsInline>` 으로 재생
- 끝까지 재생되면 자동 닫기
- 우상단 [건너뛰기] 버튼으로 즉시 종료
- 파일이 없으면 "전문가 영상 준비 중이에요" fallback 표시 (앱은 정상 동작)

## 추천 사양
- 길이: 10~20초 (한 사이클)
- 해상도: 720x1280(세로) 또는 1280x720(가로) 권장
- 코덱: H.264 (모바일 호환), AAC 오디오 또는 음소거
- 용량: 파일당 5MB 이하 권장 (PWA precache는 안 함)

## gitignore
용량이 크면 `.gitignore`에 `ui_ux/frontend/public/videos/*.mp4` 추가 고려.
지금은 파일이 없으므로 별도 처리 X.
