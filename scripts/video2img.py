import cv2
import os

# === 설정 ===
video_path = 'D:/DensityEstimationAI/datasets/test/video/test11.mp4'         # 입력 영상 경로
output_dir = 'D:/DensityEstimationAI/datasets/test/test11-img'          # 저장할 폴더
frame_interval = 1                    # 몇 프레임마다 저장할지 (1이면 모두 저장)

# === 출력 폴더 생성 ===
os.makedirs(output_dir, exist_ok=True)

# === 비디오 열기 ===
cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        output_path = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(output_path, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"총 {saved_count}장의 이미지가 저장되었습니다.")
