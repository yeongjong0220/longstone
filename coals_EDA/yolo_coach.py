import cv2
import time
import numpy as np
from ultralytics import YOLO

# 1. 2D 좌표 사이의 각도 계산 함수
def calculate_angle_2d(a, b, c):
    """ a, c: 양끝 관절(골반, 발목), b: 중심 관절(무릎) """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def main():
    video_path = 'test_video.mp4' 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 'test_video.mp4' 파일을 찾을 수 없습니다.")
        return

    print("⏳ YOLOv8-Pose 코칭 모델을 준비 중입니다...")
    model = YOLO('yolov8n-pose.pt') 
    print("✅ 코칭 시작! (종료: 'q' 키)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("동영상 재생이 끝났습니다.")
            break 
            
        frame = cv2.resize(frame, (800, 600))
        start_time = time.time()
        
        # 2. YOLOv8 추론 실행
        results = model(frame, verbose=False) 
        
        # 화면에 뼈대 렌더링 (배경)
        annotated_frame = results[0].plot()
        
        # 3. 좌표 추출 및 코칭 로직
        # 화면에 사람이 1명 이상 감지되었고, 키포인트 데이터가 있다면
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # 첫 번째 사람(0)의 키포인트(xy) 배열 가져오기
            # YOLOv8은 COCO 포맷을 사용: 11번=왼쪽 골반, 13번=왼쪽 무릎, 15번=왼쪽 발목
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            
            l_hip = keypoints[11]
            l_knee = keypoints[13]
            l_ankle = keypoints[15]
            
            # YOLO는 관절이 화면에 안 보이면 좌표를 [0, 0]으로 반환합니다.
            # 세 관절이 모두 정상적으로 화면에 잡혔을 때만 계산
            if not (np.array_equal(l_hip, [0, 0]) or np.array_equal(l_knee, [0, 0]) or np.array_equal(l_ankle, [0, 0])):
                
                # 각도 계산
                knee_angle = calculate_angle_2d(l_hip, l_knee, l_ankle)
                
                # 무릎 위치(픽셀)에 현재 각도 숫자 띄우기
                knee_x, knee_y = int(l_knee[0]), int(l_knee[1])
                cv2.putText(annotated_frame, f"{int(knee_angle)} deg", (knee_x + 10, knee_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                # 🚨 [핵심 코칭 로직] EDA에서 발굴한 기준값(160도) 적용
                if knee_angle < 160:
                    # 각도가 너무 좁으면 빨간색 경고
                    cv2.putText(annotated_frame, "WARNING: Straighten your knee!", 
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    # 자세가 좋으면 파란색 칭찬
                    cv2.putText(annotated_frame, "GOOD POSTURE", 
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

        # FPS 계산 및 출력
        elapsed_time = time.time() - start_time
        fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # 영상 출력
        cv2.imshow('AI Pilates Coach (YOLOv8)', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()