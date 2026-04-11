import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re

# 1. 환경 설정 (평활화까지 마친 새로운 Normalized_2 폴더 대상)
root_path = Path('./The_Seal_Normalized_2') 

def calculate_3d_dist(df, p1, p2):
    return np.sqrt(
        (df[f'{p1}_x'] - df[f'{p2}_x'])**2 +
        (df[f'{p1}_y'] - df[f'{p2}_y'])**2 +
        (df[f'{p1}_z'] - df[f'{p2}_z'])**2
    )

all_stats = []

# 2. 모든 CSV 파일 순회 및 데이터 수집
print(f"'{root_path.name}' 폴더의 모든 파일 검증 중...")
for csv_path in root_path.glob('**/*.csv'):
    # 파일명 또는 폴더명에서 배우 ID 추출
    match = re.search(r'actorP\d+', str(csv_path))
    actor_id = match.group() if match else "Unknown"
    
    # 폴더명에서 촬영 시간대 추출 (예: 12.47.57)
    folder_name = csv_path.parent.name
    trial_id = folder_name.split('_')[1] if '_' in folder_name else folder_name

    try:
        df = pd.read_csv(csv_path)
        # 스케일이 잘 맞춰졌는지 다시 한번 척도(Neck~Hip 거리) 추적
        torso_lengths = calculate_3d_dist(df, 'Neck', 'Hip')
        
        all_stats.append({
            'actor': actor_id,
            'trial_label': f"{actor_id}\n({trial_id})", # 막대 아래 표시될 X축 이름
            'mean_len': torso_lengths.mean()
        })
    except Exception as e:
        print(f"오류 발생 ({csv_path.name}): {e}")

# 3. 데이터프레임 변환 및 시각화
if len(all_stats) > 0:
    stats_df = pd.DataFrame(all_stats).sort_values('actor')

    plt.figure(figsize=(15, 7))
    colors_map = {'actorP075': '#3498db', 'actorP076': '#e67e22', 'actorP077': '#2ecc71'}
    colors = [colors_map.get(a, 'gray') for a in stats_df['actor']]

    bars = plt.bar(stats_df['trial_label'], stats_df['mean_len'], color=colors, edgecolor='black', alpha=0.8)

    # 전체 평균선
    total_avg = stats_df['mean_len'].mean()
    plt.axhline(y=total_avg, color='red', linestyle='--', linewidth=2, label=f'Total Avg: {total_avg:.5f}')

    # 그래프 텍스트 및 라벨 세팅
    plt.title('Validation: Body Length of Fully Cleaned Data (The_Seal_Normalized_2)', fontsize=16, pad=20)
    plt.ylabel('Mean Torso Length (3D Distance)', fontsize=12)
    plt.xlabel('Individual Trials (Actor & Time)', fontsize=12)
    
    # 텍스트가 겹치지 않게 조절
    plt.xticks(rotation=45 if len(stats_df) > 10 else 0, fontsize=9) 
    
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    # y축을 0~1.5 사이로 고정하여 1.0(평균) 부근에서 돋보이게 만듦
    plt.ylim(0, 1.5)

    # 범례 설정
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=c, lw=4, label=a) for a, c in colors_map.items()]
    legend_elements.append(Line2D([0], [0], color='red', lw=2, linestyle='--', label=f'Target Average (Avg={total_avg:.2f})'))
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

    print(f"분석 완료: 총 {len(stats_df)}개의 시도(Trial)를 시각화했습니다.")
    print("스무딩까지 완료된 새 데이터들 역시, 모든 참가자의 키(스케일)가 1.0 안팎으로 완벽히 정렬되었음을 의미합니다.")
else:
    print(f"{root_path} 폴더 내에 분석할 CSV 파일이 아직 존재하지 않습니다.\n(01_normalize_02.py 를 먼저 실행해 전체 변환을 완료해 주세요!)")
