import pandas as pd
import numpy as np
from pathlib import Path

# 1. 정규화된 데이터를 저장할 폴더 생성
output_root = Path('./The_Seal_Normalized')
output_root.mkdir(exist_ok=True)

print("전체 데이터 정규화를 시작합니다...")

# 2. 모든 CSV 파일 순회
input_root = Path('./The Seal')
csv_files = list(input_root.glob('**/*.csv'))

for csv_path in csv_files:
    # 데이터 로드
    df = pd.read_csv(csv_path)
    
    # 몸통 길이(Neck-Hip) 실시간 계산
    dist = np.sqrt(
        (df['Neck_x'] - df['Hip_x'])**2 +
        (df['Neck_y'] - df['Hip_y'])**2 +
        (df['Neck_z'] - df['Hip_z'])**2
    )
    mean_len = dist.mean()
    
    # 좌표 컬럼들 정규화 (mean_len으로 나누기)
    coord_cols = [c for c in df.columns if c.endswith(('_x', '_y', '_z'))]
    df[coord_cols] = df[coord_cols] / mean_len
    
    # 3. 저장 경로 설정 (원본 구조 유지)
    relative_path = csv_path.relative_to(input_root)
    save_path = output_root / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 정규화된 데이터 저장
    df.to_csv(save_path, index=False)

print(f"성공! 총 {len(csv_files)}개의 파일이 './The_Seal_Normalized' 폴더에 정규화되어 저장되었습니다.")