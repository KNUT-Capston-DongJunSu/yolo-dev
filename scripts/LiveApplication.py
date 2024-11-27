from ultralytics import YOLO
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import torch
from yolov5 import YOLOTrainer

class ApplicationHandler:
    def __init__(self):
        """
        Application Handler for YOLO inference with PiCamera.
        
        :param model_path: Path to the trained YOLO model.
        :param conf_threshold: Confidence threshold for predictions.
        :param iou_threshold: IoU threshold for predictions.
        """

        self.model = YOLOTrainer(weight_path="")
        
        # PiCamera 초기화
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 30
        self.rawCapture = PiRGBArray(self.camera, size=(640, 480))

    def Run(self):
        """
        Run YOLO inference in real-time using PiCamera.
        """
        print("Camera warming up...")
        time.sleep(0.1)

        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            annotated_image = self.model.predict(image=image)
            cv2.imshow("YOLO Detection", annotated_image)

            # 프레임 초기화
            self.rawCapture.truncate(0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
