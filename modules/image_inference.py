import os
import cv2 
from ultralytics import YOLO

def run_inference(model_path, img_path, output_dir):    
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(model_path)     
    results = model.predict(
        source=img_path,
        conf=0.05,
        imgsz=640,
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