import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from PIL import Image

image_folder = './datasets/test/market/'
output_video = 'output_video.mp4'

images = sorted([img for img in os.listdir(image_folder) if img.lower().endswith(".jpg")])

if not images:
    raise ValueError("이미지 없음")

# 한글 경로 대응용 이미지 로더
def read_image_unicode(path):
    with Image.open(path) as img:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

first_image_path = os.path.join(image_folder, images[0])
frame = read_image_unicode(first_image_path)
height, width, _ = frame.shape

fps = 30
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for image in images:
    img_path = os.path.join(image_folder, image)
    try:
        frame = read_image_unicode(img_path)
        out.write(frame)
    except Exception as e:
        print(f"[오류] {img_path} 읽기 실패: {e}")

out.release()
print("영상 생성 완료:", output_video)