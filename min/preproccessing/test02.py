import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

# Features 폴더 대신 Normalized 폴더의 파일을 불러오도록 경로 수정
file_path = './The_Seal_Normalized_2/actorP075/20220930_12.47.57_가산A/keypoints_필라테스_가산_A_Mat_The Seal_고급_actorP075_20220930_12.47.57.csv'
df = pd.read_csv(file_path)

# 관절 이름들(csv 헤더 기준, 소문자 k 등 정확히 매칭)
joints = ['Head', 'Neck', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 
          'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'Rknee', 'LAnkle', 'RAnkle', 'Hip']

# 연결선(뼈대) 쌍 정의 (해부학적 구조에 맞게 연결)
skeleton_edges = [
    ('Head', 'Neck'),
    ('Neck', 'LShoulder'), ('Neck', 'RShoulder'),
    ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
    ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
    ('Neck', 'Hip'),
    ('Hip', 'LHip'), ('Hip', 'RHip'),
    ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
    ('RHip', 'Rknee'), ('Rknee', 'RAnkle')
]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# 비율을 1:1:1 로 맞춰서 찌그러지지 않게 함
ax.set_box_aspect([1, 1, 1])

# 플롯 객체 초기화 (점과 선)
points_plot = ax.scatter([], [], [], c='blue', s=30)
lines_plot = [ax.plot([], [], [], c='red', linewidth=2)[0] for _ in skeleton_edges]

# 축 레이블
ax.set_xlabel('X')
ax.set_ylabel('Z (Depth)')
ax.set_zlabel('Y (Height)')
title = ax.set_title('3D Skeleton Animation (Normalized)')

# X, Y, Z 데이터 좌표 최소/최대값 미리 구하기 (그래프 스케일 고정)
x_min, x_max = df[[f'{j}_x' for j in joints]].min().min(), df[[f'{j}_x' for j in joints]].max().max()
y_min, y_max = df[[f'{j}_y' for j in joints]].min().min(), df[[f'{j}_y' for j in joints]].max().max()
z_min, z_max = df[[f'{j}_z' for j in joints]].min().min(), df[[f'{j}_z' for j in joints]].max().max()

# 맷플롯립 3D에서는 축 범위를 고정해야 재생 중에 스케일이 안 변함
ax.set_xlim(x_min - 0.2, x_max + 0.2)
ax.set_ylim(z_min - 0.2, z_max + 0.2)
ax.set_zlim(-y_max - 0.2, -y_min + 0.2) # 위아래 반전이므로 -를 붙임

def update(frame_idx):
    sample = df.iloc[frame_idx]
    
    # 각 점 좌표 가져오기 (이미지 좌표와 일치시키기 위해 Y축 반전)
    xs = [sample[f'{j}_x'] for j in joints]
    ys = [-sample[f'{j}_y'] for j in joints]
    zs = [sample[f'{j}_z'] for j in joints]
    
    # 3D 점 업데이트 (x, z, -y 순서로 세팅)
    points_plot._offsets3d = (np.array(xs), np.array(zs), np.array(ys))
    
    # 선(관절 연결) 업데이트
    for line, (j1, j2) in zip(lines_plot, skeleton_edges):
        x_coords = [sample[f'{j1}_x'], sample[f'{j2}_x']]
        y_coords = [-sample[f'{j1}_y'], -sample[f'{j2}_y']]
        z_coords = [sample[f'{j1}_z'], sample[f'{j2}_z']]
        
        line.set_data(x_coords, z_coords)
        line.set_3d_properties(y_coords)
        
    title.set_text(f'3D Skeleton Animation (Normalized) - Frame: {frame_idx}')
    return [points_plot] + lines_plot

# 밀리초 단위 딜레이 계산 (예: 30fps = 약 33ms)
ani = FuncAnimation(fig, update, frames=len(df), interval=33, blit=False)

plt.show()
