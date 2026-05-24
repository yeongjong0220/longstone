import pandas as pd
import matplotlib.pyplot as plt

# 시각화하고 싶은 파일 경로 선택
file_path = './The_Seal_Features/actorP075/20220930_12.47.57_가산A/keypoints_필라테스_가산_A_Mat_The Seal_고급_actorP075_20220930_12.47.57.csv'
df = pd.read_csv(file_path)

plt.figure(figsize=(15, 8))

# 1. 고관절 각도 그래프
plt.subplot(2, 1, 1)
plt.plot(df['L_Hip_Angle'], label='Left Hip', color='blue')
plt.plot(df['R_Hip_Angle'], label='Right Hip', color='cyan', linestyle='--')
plt.title('Hip Angles over Time')
plt.ylabel('Degrees')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 무릎 각도 그래프
plt.subplot(2, 1, 2)
plt.plot(df['L_Knee_Angle'], label='Left Knee', color='red')
plt.plot(df['R_Knee_Angle'], label='Right Knee', color='orange', linestyle='--')
plt.title('Knee Angles over Time')
plt.xlabel('Frame Index')
plt.ylabel('Degrees')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()