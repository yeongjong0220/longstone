import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
import os

# ----------------- 1. 이상치 제거 및 평활화 함수 정의 -----------------
def process_coordinate(series, window=15, poly_order=3, threshold=3):
    """
    3D 좌표의 각 축(X, Y, Z)에 대한 노이즈와 떨림을 정제하는 함수
    """
    # 1. 이상치 처리 (Rolling Median 기반)
    rolling_med = series.rolling(window=window, center=True, min_periods=1).median()
    difference = np.abs(series - rolling_med)
    median_abs_deviation = difference.rolling(window=window, center=True, min_periods=1).median()
    
    # 튀는 값 마스킹 (threshold 기준)
    outlier_idx = difference > (threshold * median_abs_deviation + 1e-5)
    
    series_clean = series.copy()
    series_clean[outlier_idx] = rolling_med[outlier_idx]
    
    # 결측치 보간
    series_clean = series_clean.interpolate()
    
    # 2. 사비츠키-골레이 필터 적용 (부드러운 곡선)
    window_length = window if window % 2 != 0 else window + 1
    if len(series_clean) > window_length:
        smoothed = savgol_filter(series_clean, window_length=window_length, polyorder=poly_order)
    else:
        smoothed = series_clean.values
        
    return smoothed

# ----------------- 2. 폴더 순회 및 일괄 처리 설정 -----------------
input_root = Path('./The Seal')
csv_files = list(input_root.glob('**/*.csv'))

# 결과를 저장할 새 폴더
output_root = Path('./The_Seal_Normalized_2')
output_root.mkdir(parents=True, exist_ok=True)

print(f"총 {len(csv_files)}개의 파일에 대한 정규화 및 정제(스무딩)를 시작합니다...")

for target_file in csv_files:
    # ----------------- 3. 스케일 정규화 (Normalization) -----------------
    df = pd.read_csv(target_file)
    
    dist = np.sqrt(
        (df['Neck_x'] - df['Hip_x'])**2 +
        (df['Neck_y'] - df['Hip_y'])**2 +
        (df['Neck_z'] - df['Hip_z'])**2
    )
    mean_len = dist.mean()
    
    # 좌표 컬럼들 정규화
    coord_cols = [c for c in df.columns if c.endswith(('_x', '_y', '_z'))]
    df[coord_cols] = df[coord_cols] / mean_len
    
    # ----------------- 4. 부드러운 스무딩 (Smoothing & Clean) -----------------
    for col in coord_cols:
        df[col] = process_coordinate(df[col], window=15, poly_order=3, threshold=3)
        
    # 기존 산출된 Angle 데이터가 있다면 동일하게 스무딩 적용
    angle_cols = [c for c in df.columns if c.endswith('_Angle')]
    for col in angle_cols:
        df[col] = process_coordinate(df[col], window=15, poly_order=3, threshold=3)
    
    # ----------------- 5. 데이터 저장 -----------------
    try:
        relative_path = target_file.relative_to(input_root)
    except ValueError:
        relative_path = Path(*target_file.parts[-4:])
        
    save_path = output_root / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(save_path, index=False)
    print(f"정제 완료: {target_file.name}")

print(f"\n완성! 모든 파일이 {output_root.absolute()} 폴더 체계 안에 정규화 및 평활화 되었습니다.")
