import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re

# 1. 환경 설정
root_path = Path('./The Seal') 

def calculate_3d_dist(df, p1, p2):
    return np.sqrt(
        (df[f'{p1}_x'] - df[f'{p2}_x'])**2 +
        (df[f'{p1}_y'] - df[f'{p2}_y'])**2 +
        (df[f'{p1}_z'] - df[f'{p2}_z'])**2
    )

all_stats = []

# 2. 모든 CSV 파일 순회 및 데이터 수집
print("모든 파일의 데이터를 개별적으로 분석 중입니다...")
for csv_path in root_path.glob('**/*.csv'):
    # 파일명 또는 폴더명에서 배우 ID 추출
    match = re.search(r'actorP\d+', str(csv_path))
    actor_id = match.group() if match else "Unknown"
    
    # 폴더명에서 촬영 시간대 추출 (예: 12.47.57)
    # 폴더 구조: .../actorP075/20220930_12.47.57_가산A/file.csv
    folder_name = csv_path.parent.name
    trial_id = folder_name.split('_')[1] if '_' in folder_name else folder_name

    try:
        df = pd.read_csv(csv_path)
        torso_lengths = calculate_3d_dist(df, 'Neck', 'Hip')
        
        all_stats.append({
            'actor': actor_id,
            'trial_label': f"{actor_id}\n({trial_id})", # 막대 아래 표시될 이름
            'mean_len': torso_lengths.mean()
        })
    except Exception as e:
        print(f"오류 발생 ({csv_path.name}): {e}")

# 데이터프레임 변환 및 정렬 (배우별로 모아서 보기 위해)
stats_df = pd.DataFrame(all_stats).sort_values('actor')

# 3. 시각화
plt.figure(figsize=(15, 7)) # 가로로 더 길게 조절
colors_map = {'actorP075': '#3498db', 'actorP076': '#e67e22', 'actorP077': '#2ecc71'}
colors = [colors_map.get(a, 'gray') for a in stats_df['actor']]

bars = plt.bar(stats_df['trial_label'], stats_df['mean_len'], color=colors, edgecolor='black', alpha=0.8)

# 전체 평균선
total_avg = stats_df['mean_len'].mean()
plt.axhline(y=total_avg, color='red', linestyle='--', linewidth=2, label=f'Total Avg: {total_avg:.3f}')

# 그래프 꾸미기
plt.title('Step 0: Body Length for Each Individual Trial', fontsize=16, pad=20)
plt.ylabel('Mean Torso Length (3D Distance)', fontsize=12)
plt.xlabel('Individual Trials (Actor & Time)', fontsize=12)
plt.xticks(rotation=0, fontsize=9) # 라벨이 겹치면 rotation=45로 변경하세요
plt.grid(axis='y', linestyle=':', alpha=0.6)

# 범례 추가 (배우별 색상 설명)
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=c, lw=4, label=a) for a, c in colors_map.items()]
legend_elements.append(Line2D([0], [0], color='red', lw=2, linestyle='--', label='Total Average'))
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()

print(f"분석 완료: 총 {len(stats_df)}개의 시도(Trial)를 시각화했습니다.")