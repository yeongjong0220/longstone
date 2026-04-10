import cv2
import time
from ultralytics import YOLO

def main():
    # 복사해둔 테스트 영상 경로
    video_path = 'test_video.mp4' 
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 'test_video.mp4' 파일을 찾을 수 없습니다.")
        return

    print("⏳ YOLOv8-Pose 모델을 로드하는 중입니다...")
    
    # [YOLOv8-Pose 초기화] - 가장 가볍고 빠른 nano 버전
    model_yolo = YOLO('yolov8n-pose.pt') 
    
    print("✅ 로드 완료! 영상을 재생합니다. (종료: 'q' 키)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("동영상 재생이 끝났습니다.")
            break 
            
        # 연산 속도 확보를 위해 프레임 리사이즈
        frame = cv2.resize(frame, (800, 600))
        
        # 🔵 YOLOv8-Pose 추론 및 시간(FPS) 측정
        start_time = time.time()
        
        # 모델 추론 (verbose=False로 터미널 출력 생략)
        results = model_yolo(frame, verbose=False) 
        
        # FPS 계산
        elapsed_time = time.time() - start_time
        fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
        
        # 결과 화면에 렌더링 (YOLO 내장 plot 함수)
        annotated_frame = results[0].plot()
        
        # 좌측 상단에 FPS 텍스트 출력
        cv2.putText(annotated_frame, f'YOLOv8n-Pose | FPS: {int(fps)}', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # 화면 출력
        cv2.imshow('YOLOv8-Pose Realtime Test', annotated_frame)
        
        # 'q' 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()