from picamera.array import PiRGBArray
from picamera import PiCamera
from scripts.yolov5 import YOLOTrainer
from scripts.density_test import Density
import cv2
import math
import time  # 실시간 시뮬레이션용

class ApplicationHandler:    
    def __init__(self, weight_path):
        # 모델 초기화
        self.model = YOLOTrainer(weight_path=weight_path)
        
        # PiCamera 초기화
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 30
        self.rawCapture = PiRGBArray(self.camera, size=(640, 480))
        self.frame_height = self.camera.resolution[1]  # 실제 카메라 해상도 사용

        # Density 초기화
        self.density_manager = Density(self.frame_height)

    def Run(self):
        print("Camera warming up...")
        time.sleep(0.1)

        for frame_idx, frame in enumerate(self.camera.capture_continuous(
            self.rawCapture, format="bgr", use_video_port=True)):
            print(f"\n=== 프레임 {frame_idx + 1} ===")
            image = frame.array

            predictions, annotated_image = self.model.predict(image=image, return_raw=True)
            cv2.imshow("YOLO Detection", annotated_image)

            # 밀도 프레임당 계산
            self.density_manager.calculate_density(predictions)

            # 프레임 초기화
            self.rawCapture.truncate(0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()

