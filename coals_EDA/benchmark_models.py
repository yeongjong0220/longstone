import cv2
import time
import mediapipe as mp
from ultralytics import YOLO

def main():
    video_path = 'test_video.mp4' 
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 'test_video.mp4' 파일을 찾을 수 없습니다.")
        return

    print("⏳ 모델을 로드하는 중입니다...")
    
    # 🌟 정상적인 MediaPipe 호출 방식
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_mp = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # YOLOv8-Pose 호출
    model_yolo = YOLO('yolov8n-pose.pt') 
    
    print("✅ 로드 완료! 비교 테스트를 시작합니다. (종료: 'q' 키)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("동영상 재생이 끝났습니다.")
            break 
            
        frame = cv2.resize(frame, (640, 480))
        frame_mp = frame.copy()
        frame_yolo = frame.copy()
        
        # 🟢 1. MediaPipe
        start_time_mp = time.time()
        image_rgb = cv2.cvtColor(frame_mp, cv2.COLOR_BGR2RGB)
        
        # 0 나누기 에러 방지
        try:
            results_mp = pose_mp.process(image_rgb)
        except Exception as e:
            print(f"MediaPipe 추론 에러: {e}")
            break
            
        elapsed_mp = time.time() - start_time_mp
        fps_mp = 1.0 / elapsed_mp if elapsed_mp > 0 else 0
        
        if results_mp and results_mp.pose_landmarks:
            mp_drawing.draw_landmarks(frame_mp, results_mp.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        cv2.putText(frame_mp, f'MediaPipe | FPS: {int(fps_mp)}', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 🔵 2. YOLOv8-Pose
        start_time_yolo = time.time()
        results_yolo = model_yolo(frame_yolo, verbose=False) 
        
        elapsed_yolo = time.time() - start_time_yolo
        fps_yolo = 1.0 / elapsed_yolo if elapsed_yolo > 0 else 0
        
        frame_yolo = results_yolo[0].plot()
        cv2.putText(frame_yolo, f'YOLOv8-Pose | FPS: {int(fps_yolo)}', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)

        # 🖥 3. 화면 출력
        combined_frame = cv2.hconcat([frame_mp, frame_yolo])
        cv2.imshow('Model Benchmark', combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()