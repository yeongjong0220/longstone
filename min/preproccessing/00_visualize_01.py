import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re # 정규표현식 추가

# 1. 환경 설정
root_path = Path('./The Seal') 

def calculate_3d_dist(df, p1, p2):
    return np.sqrt((df[f'{p1}_x'] - df[f'{p2}_x'])**2 + (df[f'{p1}_y'] - df[f'{p2}_y'])**2 + (df[f'{p1}_z'] - df[f'{p2}_z'])**2)

all_stats = []

# 모든 CSV 파일을 찾습니다.
for csv_path in root_path.glob('**/*.csv'):
    # 파일명에서 'actorPXXX' 패턴을 찾아냅니다 (폴더 깊이 상관없음)
    match = re.search(r'actorP\d+', csv_path.name)
    actor_id = match.group() if match else "Unknown"
    
    # 촬영 시간은 파일명 뒷부분에서 추출
    trial_time = "_".join(csv_path.name.split('_')[-2:]).replace('.csv', '')

    try:
        df = pd.read_csv(csv_path)
        torso_lengths = calculate_3d_dist(df, 'Neck', 'Hip')
        
        all_stats.append({
            'actor': actor_id,
            'trial': trial_time,
            'mean_len': torso_lengths.mean()
        })
    except Exception as e:
        print(f"파일 로드 중 오류 발생 ({csv_path.name}): {e}")

stats_df = pd.DataFrame(all_stats)

# 3. 시각화: 배우별 신체 크기 차이 확인
plt.figure(figsize=(12, 6))
colors = {'actorP075': 'royalblue', 'actorP076': 'orange', 'actorP077': 'green'}

for i, actor in enumerate(stats_df['actor'].unique()):
    actor_data = stats_df[stats_df['actor'] == actor]
    plt.bar(actor_data['trial'].apply(lambda x: f"{actor}\n({x[:8]})"), 
            actor_data['mean_len'], 
            color=colors.get(actor, 'gray'),
            label=actor if i == 0 or actor not in plt.gca().get_legend_handles_labels()[1] else "")

plt.axhline(y=stats_df['mean_len'].mean(), color='red', linestyle='--', label='Total Average')
plt.title('Step 0: Comparison of Mean Torso Length Across Actors and Trials')
plt.ylabel('3D Distance (Neck to Hip)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 결과 출력
print("\n[분석 결과 요약]")
print(stats_df.groupby('actor')['mean_len'].mean())