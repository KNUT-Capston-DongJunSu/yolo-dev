import torch
import torch.nn.functional as F
from torchvision import transforms
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
from models import YoloV4
from models.PostProcessor import YOLOPostProcessor

class ApplicationHandler:
    def __init__(self, model_path: str = "trained_yolov4.pth", conf_threshold: float = 0.5, iou_threshold: float = 0.4):
        # 모델 설정
        num_classes = 3  # 예: 배경 클래스 + 2개의 객체 클래스
        self.model = YoloV4(num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()  # 모델을 평가 모드로 설정
        
        # 후처리기 설정
        self.post_processor = YOLOPostProcessor(conf_threshold=conf_threshold, iou_threshold=iou_threshold, input_dim=416)

        # PiCamera 초기화
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 30
        self.rawCapture = PiRGBArray(self.camera, size=(640, 480))

        # 이미지 전처리 함수
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((416, 416)),  # 모델 입력 크기에 맞추기
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 이미지넷 정규화 값 사용
        ])

    def Run(self) -> None:
        print("Camera warming up...")
        time.sleep(0.1)

        # 프레임 스트림 시작
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            original_image = image.copy()  # 원본 이미지 저장

            # 이미지 전처리
            input_image = self.transform(image).unsqueeze(0)  # 배치 차원 추가
            input_image = input_image.to("cuda" if torch.cuda.is_available() else "cpu")

            # 모델 실행
            with torch.no_grad():
                predictions = self.model(input_image)

            # 후처리 적용
            final_boxes = self.post_processor.process_predictions(predictions, original_img_shape=image.shape[:2])

            # 결과 출력 및 시각화
            for box in final_boxes:
                x, y, w, h, conf, class_id = box[:6]
                print(f"Class: {class_id}, Confidence: {conf:.2f}, Bounding Box: ({x}, {y}, {w}, {h})")

                # 바운딩 박스 그리기
                cv2.rectangle(original_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.putText(original_image, f"Class: {class_id}, Conf: {conf:.2f}", (int(x), int(y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 결과 디스플레이
            cv2.imshow("YOLOv4 Detection", original_image)

            # 프레임 초기화
            self.rawCapture.truncate(0)

            # 종료 조건
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ApplicationHandler(model_path="trained_yolov4.pth")
    app.Run()
