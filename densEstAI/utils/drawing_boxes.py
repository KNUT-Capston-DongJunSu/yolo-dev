import cv2 

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