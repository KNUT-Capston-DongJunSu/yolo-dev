import os
import cv2 
import glob
import numpy as np
from ultralytics import YOLO
from collections import deque
from PIL import Image, ImageDraw

track_hist = {}

def filter_tracks_by_class(tracks):
    # tracks: 리스트 of dict {id, x, y, w, h} 현재 프레임 정보
    filtered_ids = []
    for obj in tracks:
        track_id = obj[4]
        cx = (obj[0] + obj[2]) / 2
        cy = (obj[1] + obj[3]) / 2

        if track_id not in track_hist:
            track_hist[track_id] = deque(maxlen=10)
        track_hist[track_id].append((cx, cy))

        # 이동 거리 계산
        if len(track_hist[track_id]) >= 2:
            dist_sum = 0
            pts = list(track_hist[track_id])
            for i in range(len(pts)-1):
                dist_sum += ((pts[i+1][0] - pts[i][0])**2 + (pts[i+1][1] - pts[i][1])**2) ** 0.5
            avg_dist = dist_sum / (len(pts)-1)
        else:
            avg_dist = 0

        # 임계치 비교
        if avg_dist > 0.1:  # 움직임 있으면 사람으로 간주
            filtered_ids.append(track_id)
        else:
            # 움직임 적으면 오탐 가능성 높음 → 제외하거나 별도 처리
            pass

    return np.array(filtered_ids).astype(int)


def draw_tracking_boxes(frame, tracked_objects):
    """Bounding box와 트래킹 ID 표시"""
    original_img = frame.copy()
    height, width = original_img.shape[:2]

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id, *rest  = map(int, obj)
        cv2.rectangle(
            original_img, 
            (x1, y1), (x2, y2), 
            color=(255, 0, 0), 
            thickness=2
            )
        cv2.putText(
            original_img, 
            f"ID: {track_id}", 
            (x1, y1), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.3, (0, 0, 0), 1
            )

    cv2.putText(
        original_img, 
        f"{len(tracked_objects)} people", 
        (width - 200, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, (255, 255, 255), 2
        )
    
    return original_img

def inference_image(model_path, img_path, output_dir):    
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(model_path)     
    results = model.predict(
        source=img_path,
        conf=0.05,
        imgsz=1280,
        save=False,          # 저장 안 하고 메모리에서 직접 처리
        classes=[0],         # 클래스 0만
        exist_ok=True
    )

    for res in results:
        img_path = res.path  # 이미지 경로
        original_img = cv2.imread(img_path)
        height, width = original_img.shape[:2]

        boxes = res.boxes.xyxy.cpu().numpy().astype(int)  # 전체 박스 좌표 (N,4)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(original_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        cv2.putText(original_img, f"{len(boxes)} people", (width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

        output_filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, output_filename)

        cv2.imwrite(output_path, original_img)

def tracking_object(tracker, tracker_input, frame_id):
    if tracker_input.shape[0] == 0:
        tracked_objects = []  # 또는 빈 텐서 등, 트래커가 처리할 수 없는 빈 입력에 대비
    else:
        tracked_objects = tracker.update(tracker_input, frame_id)
        filtered_ids = filter_tracks_by_class(tracked_objects)
        if len(filtered_ids) != 0:
            tracked_ids = tracked_objects[:, 4].astype(int)
            indices = np.where(np.isin(tracked_ids, filtered_ids))[0]
            tracked_objects = tracked_objects[indices] 

    return tracked_objects

def img_shape(image_path):
    img = Image.open(image_path)
    width, height = img.size
    print(f"이미지 크기: {width}x{height}")

def get_best_model(weights_dir):
    best_model = os.path.join(weights_dir, "best.pt")
    return best_model if os.path.exists(best_model) else None

def detect_display():
    return "DISPLAY" in os.environ and os.environ["DISPLAY"]