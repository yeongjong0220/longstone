# import os
# import shutil
# from glob import glob

# def main():
#     # 데이터가 있는 최상위 경로
#     base_path = '/Users/ochaemin/dev/capstone/216.필라테스 동작 데이터'
    
#     print("🔍 동영상(.mp4) 파일을 탐색 중입니다...")
    
#     # mp4 확장자를 가진 모든 파일을 찾습니다.
#     video_files = glob(os.path.join(base_path, '**/*.mp4'), recursive=True)
    
#     if video_files:
#         # 찾은 영상 중 첫 번째 영상을 타겟으로 지정합니다.
#         target_video = video_files[0]
#         print(f"\n✅ 영상을 찾았습니다!\n원본 경로: {target_video}")
        
#         # 현재 폴더(EDA)에 'test_video.mp4'라는 이름으로 복사합니다.
#         dest_path = 'test_video.mp4'
#         print(f"\n⏳ '{dest_path}' 파일로 복사를 시작합니다...")
        
#         shutil.copy(target_video, dest_path)
#         print("🎉 복사 완료! 이제 벤치마크 테스트를 실행할 수 있습니다.")
#     else:
#         print("\n❌ mp4 파일을 찾을 수 없습니다. (데이터 다운로드 시 원천데이터가 포함되었는지 확인해주세요)")

# if __name__ == "__main__":
#     main()

import os
import shutil
from glob import glob

def main():
    # 데이터가 있는 최상위 경로
    base_path = '/Users/ochaemin/dev/capstone/216.필라테스 동작 데이터'
    
    print("🔍 측면(camera3) 동영상 파일을 탐색 중입니다...")
    
    # 🌟 핵심 수정: 파일 경로에 'camera3'가 포함된 mp4 파일만 찾습니다!
    video_files = glob(os.path.join(base_path, '**/*camera3*/**/*.mp4'), recursive=True)
    
    if not video_files: # 만약 camera3 폴더 방식이 아니라 파일명에 CAM_3이 있다면
        video_files = glob(os.path.join(base_path, '**/*CAM_3*.mp4'), recursive=True)

    if video_files:
        target_video = video_files[0]
        print(f"\n✅ 측면 영상을 찾았습니다!\n원본 경로: {target_video}")
        
        dest_path = 'test_video.mp4'
        print(f"\n⏳ '{dest_path}' 파일로 덮어쓰기를 시작합니다...")
        
        shutil.copy(target_video, dest_path)
        print("🎉 복사 완료! 이제 코칭 AI(yolo_coach.py)를 다시 실행해 보세요.")
    else:
        print("\n❌ 측면(camera3) mp4 파일을 찾을 수 없습니다. 폴더 구조를 확인해 주세요.")

if __name__ == "__main__":
    main()