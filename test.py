import cv2, os
from PIL import Image
from ultralytics import YOLO
from src import train_yolo
from src import VideoStreamHandler

def test(model_path, img_path, output_dir):    
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

if __name__=="__main__":
    # test(
    #     # "results/train/weights/best.pt",
    #     "SCUT_HEAD.pt",
    #     "datasets/test/SCUT-HEAD", 
    #     "results/predict/SCUT-HEAD-self"
    #     )

    # train_yolo(
    #     model_path="yolov8s.pt",
    #     config_path="configs/custom.yaml",
    #     epochs=5,
    #     imgsz=640,
    #     batch=12,
    #     resume=False, 
    #     project="results",
    #     name="train15"
    #     )
        
    video_path = "datasets/test/video/test1132.mp4"
    model_path = "results/train/weights/best.pt"
    # model_path = "SCUT_HEAD.pt"
    output_video = "results/predict/video/our_model_demo3.mp4"

    video_handler = VideoStreamHandler(video_path, model_path, output_video, 1)
    video_handler.start_stream(3)