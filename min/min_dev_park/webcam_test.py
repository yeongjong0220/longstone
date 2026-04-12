import cv2
import mediapipe as mp

print("="*50)
print("📸 [1단계 테스트] 웹캠 및 MediaPipe 구동 테스트")
print("- 모델(LLM) 파일 로딩 없이 오직 카메라와 뼈대(Pose)만 가볍게 확인합니다.")
print("- 'q' 키: 프로그램 종료")
print("="*50)

# ==========================================
# 1. MediaPipe 초기화
# ==========================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# 2. 실시간 웹캠 루프
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 웹캠(0번 카메라)을 열 수 없습니다. 카메라가 연결되어 있는지 확인해주세요.")
    exit()

print("✅ 웹캠 정상 연결됨! 테스트를 시작합니다...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        print("❌ 프레임을 읽어올 수 없습니다.")
        break

    # 거울 모드 (좌우 반전) 및 BGR -> RGB 변환
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe 성능 향상을 위해 이미지 쓰기 불가 설정
    image.flags.writeable = False
    
    # 뼈대 추론
    results = pose.process(image)
    
    # 다시 쓰기 가능하게 변경 및 RGB -> BGR 변환 (출력용)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 뼈대(Landmarks) 화면에 그리기
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        status_text = "Pose Detected!"
        color = (0, 255, 0)
    else:
        status_text = "Searching..."
        color = (0, 0, 255)

    cv2.putText(image, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 결과 화면 출력
    cv2.imshow('Webcam & MediaPipe Test', image)
    
    # 'q' 키 입력 시 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("👋 테스트를 종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()
