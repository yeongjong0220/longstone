import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm

# 맥북(Mac) 그래프 한글 깨짐 방지 설정
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 3D 벡터 사이의 각도 계산 함수 (제2코사인 법칙 활용)
def calculate_angle_3d(p1, p2, p3):
    """ p1, p3: 양끝 관절, p2: 중심 관절 """
    ba = p1 - p2
    bc = p3 - p2
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def main():
    base_path = '/Users/ochaemin/dev/capstone/216.필라테스 동작 데이터'
    csv_files = glob(os.path.join(base_path, '**/*.csv'), recursive=True)
    
    # 분석 속도를 위해 500개 샘플링 (전체 분석 시 [:500] 제거)
    sample_files = csv_files[:500] 

    # 전신 관절 및 파생 지표를 저장할 딕셔너리
    angles_data = {
        '팔꿈치_각도(Elbow)': [],
        '어깨_각도(Shoulder)': [],
        '골반_각도(Hip)': [],
        '무릎_각도(Knee)': [],
        '척추_굴곡도(Spine_Flexion)': [],    # 추가 1: 목-골반-무릎 각도
        '팔_수평오차(Arm_Horizontal)': [] # 추가 2: 어깨와 손목의 Y좌표(높이) 차이
    }

    print(f"총 {len(sample_files)}개의 샘플에서 [전신 키포인트 & 파생 지표]를 추출합니다...")

    for file_path in tqdm(sample_files, desc="데이터 추출 중"):
        df = pd.read_csv(file_path)
        if len(df) == 0: continue
        
        # 중간 프레임 추출
        mid_idx = len(df) // 2
        row = df.iloc[mid_idx]
        
        try:
            # 기본 좌표 셋업 (왼쪽 기준)
            neck = np.array([row['Neck_x'], row['Neck_y'], row['Neck_z']])
            shoulder = np.array([row['LShoulder_x'], row['LShoulder_y'], row['LShoulder_z']])
            elbow = np.array([row['LElbow_x'], row['LElbow_y'], row['LElbow_z']])
            wrist = np.array([row['LWrist_x'], row['LWrist_y'], row['LWrist_z']])
            hip = np.array([row['LHip_x'], row['LHip_y'], row['LHip_z']])
            knee = np.array([row['LKnee_x'], row['LKnee_y'], row['LKnee_z']])
            ankle = np.array([row['LAnkle_x'], row['LAnkle_y'], row['LAnkle_z']])
            
            # 1. 기존 4대 관절 각도 계산
            angles_data['팔꿈치_각도(Elbow)'].append(calculate_angle_3d(shoulder, elbow, wrist))
            angles_data['어깨_각도(Shoulder)'].append(calculate_angle_3d(hip, shoulder, elbow))
            angles_data['골반_각도(Hip)'].append(calculate_angle_3d(shoulder, hip, knee))
            angles_data['무릎_각도(Knee)'].append(calculate_angle_3d(hip, knee, ankle))
            
            # 2. [신규] 척추 굴곡도 (Spine Flexion)
            # 목 - 골반 - 무릎의 각도를 통해 척추가 얼마나 앞으로 숙여졌는지 파악
            angles_data['척추_굴곡도(Spine_Flexion)'].append(calculate_angle_3d(neck, hip, knee))
            
            # 3. [신규] 팔 수평도 (Arm Horizontal Alignment)
            # 어깨와 손목의 Y축(높이) 좌표 차이의 절댓값. 0에 가까울수록 지면과 평행.
            horizontal_diff = abs(row['LShoulder_y'] - row['LWrist_y'])
            angles_data['팔_수평오차(Arm_Horizontal)'].append(horizontal_diff)
            
        except KeyError:
            continue

    df_angles = pd.DataFrame(angles_data)

    print("\n" + "="*60)
    print("📊 [최종 보고용] 전신 키포인트 통계 (평균, 분산, 표준편차)")
    print("="*60)
    stats_df = df_angles.agg(['mean', 'var', 'std']).round(3).T
    stats_df.columns = ['평균(Mean)', '분산(Variance)', '표준편차(Std)']
    print(stats_df)
    print("="*60 + "\n")

    # --- [시각화: 상관관계 히트맵 및 박스플롯] ---
    plt.figure(figsize=(18, 7))

    # 1. 박스플롯 (Boxplot) - 지표별 스케일이 다르므로 표준화하여 시각화
    plt.subplot(1, 2, 1)
    # 수평오차는 각도(도)가 아니라 비율이므로 스케일이 달라 박스플롯에서 제외하거나 따로 그리는 것이 좋습니다.
    # 여기서는 보기 좋게 각도 데이터 5개만 박스플롯으로 그립니다.
    angle_cols = [col for col in df_angles.columns if '각도' in col or '굴곡도' in col]
    sns.boxplot(data=df_angles[angle_cols], palette="Set3")
    plt.title('전신 각도 및 척추 굴곡도 분포', fontsize=14)
    plt.ylabel('각도 (도)')
    plt.xticks(rotation=15)

    # 2. 상관관계 히트맵 (전체 6개 지표 모두 포함)
    plt.subplot(1, 2, 2)
    corr_matrix = df_angles.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title('파생 지표 포함 상관관계 (Correlation Matrix)', fontsize=14)

    plt.tight_layout()
    plt.savefig('Advanced_EDA_Report_v2.png')
    print("✅ 시각화 완료! 'Advanced_EDA_Report_v2.png' 파일이 생성되었습니다.")

if __name__ == "__main__":
    main()