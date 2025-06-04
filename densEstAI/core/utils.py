import cv2
import numpy as np
from collections import deque
from PIL import Image, ImageDraw

track_hist = {}

def update_tracks(tracks):
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


def custom_plot(frame, tracked_objects):
    """Bounding box와 트래킹 ID 표시"""
    original_img = frame.copy()
    height, width = original_img.shape[:2]

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id, *rest  = map(int, obj)
        cv2.rectangle(
            original_img, 
            (x1, y1), (x2, y2), 
            color=(0, 0, 255), 
            thickness=2)
        cv2.putText(
            original_img, 
            f"ID: {track_id}", 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, (0, 0, 255), 2
            )

    cv2.putText(
        original_img, 
        f"{len(tracked_objects)} people", 
        (width - 200, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, (255, 255, 255), 4
        )
    
    return original_img
