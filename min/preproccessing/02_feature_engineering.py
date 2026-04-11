import pandas as pd
import numpy as np
from pathlib import Path

# 1. 경로 설정
input_root = Path('./The_Seal_Normalized')
output_root = Path('./The_Seal_Features')
output_root.mkdir(exist_ok=True)

def get_angle_3d(v1, v2):
    """두 3D 벡터 사이의 각도를 구합니다."""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

def extract_features(df):
    # 관절 포인트 정의 (민혁님 CSV 컬럼명 기준 - Rknee 소문자 주의!)
    joints = {
        'L_Hip_Angle': ['LShoulder', 'LHip', 'LKnee'],
        'R_Hip_Angle': ['RShoulder', 'RHip', 'Rknee'],
        'L_Knee_Angle': ['LHip', 'LKnee', 'LAnkle'],
        'R_Knee_Angle': ['RHip', 'Rknee', 'RAnkle']
    }
    
    for angle_name, (p1, p2, p3) in joints.items():
        angles = []
        for i in range(len(df)):
            # 벡터 계산: p2->p1, p2->p3
            v_ba = np.array([df.at[i, f'{p1}_x'], df.at[i, f'{p1}_y'], df.at[i, f'{p1}_z']]) - \
                   np.array([df.at[i, f'{p2}_x'], df.at[i, f'{p2}_y'], df.at[i, f'{p2}_z']])
            v_bc = np.array([df.at[i, f'{p3}_x'], df.at[i, f'{p3}_y'], df.at[i, f'{p3}_z']]) - \
                   np.array([df.at[i, f'{p2}_x'], df.at[i, f'{p2}_y'], df.at[i, f'{p2}_z']])
            
            angles.append(get_angle_3d(v_ba, v_bc))
        df[angle_name] = angles

    # 3. 1차 이상치 제거: 물리적으로 불가능하거나 인식이 끊긴 경우 (예: 각도가 딱 0이거나 180인 경우)
    # 필라테스 동작 중 관절이 완전히 0도가 되는 일은 거의 없으므로 필터링
    df = df[(df['L_Knee_Angle'] > 5) & (df['L_Knee_Angle'] < 175)]
    
    return df

print("각도 데이터 추출 및 1차 정제를 시작합니다...")

for csv_path in input_root.glob('**/*.csv'):
    df = pd.read_csv(csv_path)
    df_featured = extract_features(df)
    
    # 저장
    relative_path = csv_path.relative_to(input_root)
    save_path = output_root / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_featured.to_csv(save_path, index=False)

print(f"완료! './The_Seal_Features' 폴더에 각도 데이터가 포함된 CSV들이 저장되었습니다.")