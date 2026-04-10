import os
import pandas as pd
from glob import glob

base_path = '/Users/ochaemin/dev/capstone/216.필라테스 동작 데이터'

# 1. 이번에는 json이 아니라 .csv 파일을 찾습니다.
print("CSV 키포인트 데이터를 탐색 중입니다...")
csv_files = glob(os.path.join(base_path, '**/*.csv'), recursive=True)

if csv_files:
    sample_csv = csv_files[0]
    print(f"✅ 샘플 CSV 탐색 완료: {os.path.basename(sample_csv)}\n")
    
    # 2. pandas로 CSV 파일 읽기
    df = pd.read_csv(sample_csv)
    
    # 3. 컬럼 이름들 출력 (관절 이름 확인)
    print("=== [CSV 컬럼(관절) 목록] ===")
    print(df.columns.tolist()[:30]) # 너무 길 수 있으니 30개만 출력
    if len(df.columns) > 30:
        print("... (이하 생략)")
        
    # 4. 실제 데이터 첫 3줄 출력
    print("\n=== [데이터 샘플 (첫 3 프레임)] ===")
    print(df.head(3))
else:
    print("❌ CSV 파일을 찾을 수 없습니다.")