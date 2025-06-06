import os
import cv2 
import glob
import numpy as np
from ultralytics import YOLO
from collections import deque
from PIL import Image, ImageDraw


def filter_tracks_by_class(track_hist, tracks):
    # tracks: 리스트 of [x1, y1, x2, y2, id]
    filtered_ids = []

    for obj in tracks:
        x1, y1, x2, y2, track_id, *rest = obj
        bbox = [x1, y1, x2, y2]

        # 기록 저장
        if track_id not in track_hist:
            track_hist[track_id] = deque(maxlen=10)
        track_hist[track_id].append(bbox)

        # 이동 거리 계산
        if len(track_hist[track_id]) >= 2:
            dist_sum = 0
            boxes = list(track_hist[track_id])
            for i in range(len(boxes)-1):
                cx1 = (boxes[i][0] + boxes[i][2]) / 2
                cy1 = (boxes[i][1] + boxes[i][3]) / 2
                cx2 = (boxes[i+1][0] + boxes[i+1][2]) / 2
                cy2 = (boxes[i+1][1] + boxes[i+1][3]) / 2
                dist = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
                dist_sum += dist
            avg_dist = dist_sum / (len(boxes)-1)
        else:
            avg_dist = 0

        if avg_dist > 0.1:
            filtered_ids.append(track_id)
        else:
            pass  # 오탐 제거 또는 무시

    return np.array(filtered_ids).astype(int)

def tracking_object(tracker, tracker_input, frame_id):
    print(tracker_input)
    if len(tracker_input) == 0:
        tracked_objects = []  # 또는 빈 텐서 등, 트래커가 처리할 수 없는 빈 입력에 대비
    else:
        tracked_objects = tracker.update(tracker_input, frame_id)
        print(tracked_objects)
        filtered_ids = filter_tracks_by_class(tracked_objects)
        print(tracked_objects)
        if len(filtered_ids) != 0:
            tracked_ids = tracked_objects[:, 4].astype(int)
            indices = np.where(np.isin(tracked_ids, filtered_ids))[0]
            tracked_objects = tracked_objects[indices] 

    return tracked_objects

def transform_yolo2track(track_data, maxlen=10):
    results = {}
    for i, obj in enumerate(track_data):
        x1, y1, x2, y2, *rest = obj
        bbox = [x1, y1, x2, y2]
        track_id = len(track_data)-i
        if track_id not in results:
            results[track_id] = deque(maxlen=maxlen)

        results[track_id].append(bbox)
    return results

def draw_tracking_boxes(frame, tracked_objects):
    """Bounding box와 트래킹 ID 표시"""
    if len(tracked_objects) == 0:
        return frame
    
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
        (width-200, height-30), 
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

def img_shape(image_path):
    img = Image.open(image_path)
    width, height = img.size
    print(f"이미지 크기: {width}x{height}")

def get_best_model(weights_dir):
    best_model = os.path.join(weights_dir, "best.pt")
    return best_model if os.path.exists(best_model) else None

def detect_display():
    return "DISPLAY" in os.environ and os.environ["DISPLAY"]