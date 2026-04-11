import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 시각화하고 싶은 파일 경로 선택
file_path = './The_Seal_Features/actorP075/20220930_12.47.57_가산A/keypoints_필라테스_가산_A_Mat_The Seal_고급_actorP075_20220930_12.47.57.csv'
df = pd.read_csv(file_path)

def process_signal(series, window=15, poly_order=3, threshold=3):
    """
    Rolling Median을 이용한 이상치(자잘하게 튀는 값) 제거 후,
    Savitzky-Golay 필터로 곡선을 부드럽게 평활화(Smoothing)합니다.
    """
    # 1. 이상치 처리 (Rolling Median 기반)
    rolling_med = series.rolling(window=window, center=True, min_periods=1).median()
    difference = np.abs(series - rolling_med)
    median_abs_deviation = difference.rolling(window=window, center=True, min_periods=1).median()
    
    # 이상치 마스킹 (threshold 기준)
    outlier_idx = difference > (threshold * median_abs_deviation + 1e-5)
    
    # 이상치를 rolling median으로 보정
    series_clean = series.copy()
    series_clean[outlier_idx] = rolling_med[outlier_idx]
    
    # 보간법으로 결측치가 혹시 있다면 채움
    series_clean = series_clean.interpolate()
    
    # 2. 사비츠키-골레이 필터 적용
    window_length = window if window % 2 != 0 else window + 1  # 필터의 윈도우 길이는 항상 홀수
    # 데이터 길이가 window_length 보다 짧은 경우를 대비한 예외 처리
    if len(series_clean) > window_length:
        smoothed = savgol_filter(series_clean, window_length=window_length, polyorder=poly_order)
    else:
        smoothed = series_clean.values
        
    return smoothed

# 적용할 각도 컬럼들
angles = ['L_Hip_Angle', 'R_Hip_Angle', 'L_Knee_Angle', 'R_Knee_Angle']
for col in angles:
    df[col + '_smoothed'] = process_signal(df[col], window=15, poly_order=3)

# 그래프 시각화
plt.figure(figsize=(15, 8))

# 1. 고관절 각도 그래프
plt.subplot(2, 1, 1)
plt.plot(df['L_Hip_Angle'], color='blue', alpha=0.15, label='L Hip (Raw)')
plt.plot(df['R_Hip_Angle'], color='cyan', alpha=0.15, label='R Hip (Raw)')
plt.plot(df['L_Hip_Angle_smoothed'], label='L Hip (Smoothed)', color='blue', linewidth=2)
plt.plot(df['R_Hip_Angle_smoothed'], label='R Hip (Smoothed)', color='cyan', linestyle='--', linewidth=2)
plt.title('Hip Angles over Time (Smoothed with Savitzky-Golay)')
plt.ylabel('Degrees')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 무릎 각도 그래프
plt.subplot(2, 1, 2)
plt.plot(df['L_Knee_Angle'], color='red', alpha=0.15, label='L Knee (Raw)')
plt.plot(df['R_Knee_Angle'], color='orange', alpha=0.15, label='R Knee (Raw)')
plt.plot(df['L_Knee_Angle_smoothed'], label='L Knee (Smoothed)', color='red', linewidth=2)
plt.plot(df['R_Knee_Angle_smoothed'], label='R Knee (Smoothed)', color='orange', linestyle='--', linewidth=2)
plt.title('Knee Angles over Time (Smoothed with Savitzky-Golay)')
plt.xlabel('Frame Index')
plt.ylabel('Degrees')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()