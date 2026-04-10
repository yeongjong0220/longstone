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
    """ p1, p3: 양끝 관절, p2: 중심 관절 (예: 어깨-팔꿈치-손목) """
    ba = p1 - p2
    bc = p3 - p2
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def main():
    base_path = '/Users/ochaemin/dev/capstone/216.필라테스 동작 데이터'
    print("🔍 CSV 파일 목록을 수집 중입니다...")
    csv_files = glob(os.path.join(base_path, '**/*.csv'), recursive=True)
    print(f"총 {len(csv_files)}개의 CSV 파일을 찾았습니다.")

    # EDA 결과를 담을 딕셔너리
    eda_results = {
        'class_distribution': {},
        'label_ratio': {'정상(Normal)': 0, '오답(Abnormal)': 0},
        'elbow_angles': [], # 팔꿈치 각도 샘플 (LShoulder - LElbow - LWrist)
        'shoulder_hip_ratios': [] # 어깨너비 대비 골반너비 비율 샘플
    }

    # 테스트를 위해 우선 500개의 파일만 샘플링하여 분석 (전체 분석 시 [:500] 제거)
    sample_files = csv_files[:500] 

    for file_path in tqdm(sample_files, desc="데이터 분석 중"):
        # --- [1] 메타데이터 추출 (경로 및 파일명 기반) ---
        file_name = os.path.basename(file_path)
        
        # 동작 이름 추출 (예: Mat_Spine Stretch)
        # 파일명 구조에 따라 Split 하여 동작명(Class)을 가져옵니다.
        try:
            action_class = file_name.split('_')[4] + "_" + file_name.split('_')[5]
        except:
            action_class = "Unknown"
            
        eda_results['class_distribution'][action_class] = eda_results['class_distribution'].get(action_class, 0) + 1
        
        # 정상/오답 여부 추출 (보통 파일명이나 경로에 normal/abnormal 포함)
        # 만약 경로에 'abnormal'이나 '오류' 등이 있다면 오답으로 분류
        if 'abnormal' in file_path.lower() or '오답' in file_path:
            eda_results['label_ratio']['오답(Abnormal)'] += 1
        else:
            eda_results['label_ratio']['정상(Normal)'] += 1

        # --- [2] 프레임별 각도 및 비율 계산 ---
        df = pd.read_csv(file_path)
        
        # 데이터가 너무 길면 연산이 오래 걸리므로 중간 프레임 1개만 샘플링
        if len(df) == 0: continue
        mid_idx = len(df) // 2
        row = df.iloc[mid_idx]
        
        try:
            # 왼쪽 팔꿈치 각도 계산 (LShoulder, LElbow, LWrist)
            p1 = np.array([row['LShoulder_x'], row['LShoulder_y'], row['LShoulder_z']])
            p2 = np.array([row['LElbow_x'], row['LElbow_y'], row['LElbow_z']])
            p3 = np.array([row['LWrist_x'], row['LWrist_y'], row['LWrist_z']])
            angle = calculate_angle_3d(p1, p2, p3)
            eda_results['elbow_angles'].append(angle)
            
            # 비율 계산: 어깨 너비 vs 골반 너비 비율
            shoulder_dist = np.linalg.norm(p1 - np.array([row['RShoulder_x'], row['RShoulder_y'], row['RShoulder_z']]))
            hip_dist = np.linalg.norm(np.array([row['LHip_x'], row['LHip_y'], row['LHip_z']]) - 
                                      np.array([row['RHip_x'], row['RHip_y'], row['RHip_z']]))
            ratio = hip_dist / (shoulder_dist + 1e-6)
            eda_results['shoulder_hip_ratios'].append(ratio)
        except KeyError:
            # 컬럼명이 다른 경우 패스
            continue

    # --- [3] 결과 시각화 (그래프 그리기) ---
    plt.figure(figsize=(15, 10))

    # 1. 정상/오답 비율 (Pie Chart)
    plt.subplot(2, 2, 1)
    labels = eda_results['label_ratio'].keys()
    sizes = eda_results['label_ratio'].values()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#F44336'])
    plt.title('정상 vs 오답 데이터 비율')

    # 2. 클래스 분포 (Bar Chart)
    plt.subplot(2, 2, 2)
    classes = list(eda_results['class_distribution'].keys())
    counts = list(eda_results['class_distribution'].values())
    sns.barplot(x=counts, y=classes, palette='viridis')
    plt.title('동작(Class)별 데이터 수 분포')
    plt.xlabel('클립 수')

    # 3. 왼쪽 팔꿈치 각도 분포 (Histogram)
    plt.subplot(2, 2, 3)
    sns.histplot(eda_results['elbow_angles'], bins=30, kde=True, color='skyblue')
    plt.title('샘플 프레임의 왼쪽 팔꿈치 각도 분포')
    plt.xlabel('각도 (도)')

    # 4. 신체 비율 (골반너비/어깨너비) 분포 (Histogram)
    plt.subplot(2, 2, 4)
    sns.histplot(eda_results['shoulder_hip_ratios'], bins=30, kde=True, color='salmon')
    plt.title('신체 비율 (골반너비 / 어깨너비) 분포')
    plt.xlabel('비율')

    plt.tight_layout()
    plt.savefig('EDA_Summary_Report.png') # 이미지로 저장
    print("✅ 분석 완료! 'EDA_Summary_Report.png' 파일이 생성되었습니다. 팀장님께 공유해주세요!")

if __name__ == "__main__":
    main()