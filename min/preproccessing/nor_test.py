import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 파일 경로 설정
raw_path = './The Seal/고급/actorP075/20220930_12.47.57_가산A/keypoints_필라테스_가산_A_Mat_The Seal_고급_actorP075_20220930_12.47.57.csv'
nor_path = './The_Seal_Normalized/고급/actorP075/20220930_12.47.57_가산A/keypoints_필라테스_가산_A_Mat_The Seal_고급_actorP075_20220930_12.47.57.csv'

try:
    df_raw = pd.read_csv(raw_path)
    df_nor = pd.read_csv(nor_path)
except FileNotFoundError:
    # 혹시 경로 구조(고급 폴더 유무 등)가 다를 경우 대비 다른 경로 시도
    raw_path = './The Seal/actorP075/20220930_12.47.57_가산A/keypoints_필라테스_가산_A_Mat_The Seal_고급_actorP075_20220930_12.47.57.csv'
    nor_path = './The_Seal_Normalized/actorP075/20220930_12.47.57_가산A/keypoints_필라테스_가산_A_Mat_The Seal_고급_actorP075_20220930_12.47.57.csv'
    df_raw = pd.read_csv(raw_path)
    df_nor = pd.read_csv(nor_path)

# 몸통 길이 (Neck - Hip) 거리 계산 함수
def get_torso_length(df):
    return np.sqrt(
        (df['Neck_x'] - df['Hip_x'])**2 +
        (df['Neck_y'] - df['Hip_y'])**2 +
        (df['Neck_z'] - df['Hip_z'])**2
    )

raw_len = get_torso_length(df_raw)
nor_len = get_torso_length(df_nor)

# ----------------- 그래프 1: 정규화 증명 (몸통 스케일) -----------------
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(raw_len, label='Original Torso Length', color='gray')
plt.plot(nor_len, label='Normalized Torso Length (Target=1.0)', color='red', linewidth=2)
plt.axhline(y=1.0, color='blue', linestyle='--', label='1.0 Reference Line')
plt.title('Validation of Scale Normalization (Neck to Hip Distance Over Time)')
plt.ylabel('Distance in 3D Space')
plt.xlabel('Frame')
plt.legend()
plt.grid(True, alpha=0.3)

# ----------------- 그래프 2: 노이즈(튀는 부분) 비교 (발목 Z 궤적) -----------------
plt.subplot(2, 1, 2)
# 원래 노이즈는 형태 자체이기 때문에 스케일링을 거쳐도 그대로 남아있음을 증명하는 그래프
plt.plot(df_raw['LAnkle_z'] / df_raw['LAnkle_z'].abs().max(), label='Original Left Ankle Z (Scaled down for visual match)', color='gray', alpha=0.7)
plt.plot(df_nor['LAnkle_z'] / df_nor['LAnkle_z'].abs().max(), label='Normalized Left Ankle Z (Scaled down for visual match)', color='red', linestyle='--')
plt.title('Why Normalization does NOT remove Spikes (Jitter comparison)')
plt.ylabel('Normalized Trajectory')
plt.xlabel('Frame')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
