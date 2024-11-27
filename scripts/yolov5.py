import torch
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

class YOLOTrainer:
    def __init__(self, model_path: str = None, config_path: str = None, weight_path: str = None,  device: str = None):
        """
        YOLO Trainer Class to handle training, validation, and inference.

        :param model_path: Path to the pre-trained YOLO model (e.g., 'models/yolov5s.pt').
        :param config_path: Path to the YOLO configuration file (e.g., 'configs/custom.yaml').
        :param device: Device to use ('cuda' or 'cpu'). Automatically detected if None.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.config_path = config_path
        self.train_model = None  # 훈련에 사용될 모델
        self.predict_model = None  # 예측에 사용될 모델
        print(f"Initialized YOLO on device: {self.device}")

    def load_model(self, mode: str):
        if mode == 'train':
            self.train_model = YOLO(self.model_path).to(self.device)
        elif mode == 'predict':
            if self.weight_path is None:
                raise ValueError("weight_path is required for 'predict' mode.")
            self.predict_model = YOLO(self.weight_path).to(self.device)
        else:
            raise ValueError("Invalid mode. Use 'train' or 'predict'.")
        
    def train(self, epochs: int = 100, imgsz: int = 256, batch_size: int = 8, project: str = 'results', name: str = 'train'):
        """
        Train the YOLO model.
        
        :param epochs: Number of training epochs.
        :param imgsz: Image size for training.
        :param batch_size: Batch size for training.
        :param project: Directory to save training results.
        :param name: Name of the training session.
        """
        if not self.train_model:
            self.load_model(mode='train')
        self.model.train(
            data=self.config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=self.device,
            project=project,
            name=name
        )
        print(f"Training completed. Results saved in {project}/{name}")

    def validate(self, imgsz: int = 256):
        """
        Validate the YOLO model.
        
        :param imgsz: Image size for validation.
        :return: Validation metrics (e.g., mAP, precision, recall).
        """
        if not self.train_model:
            self.load_model(mode='train')
        metrics = self.model.val(
            data=self.config_path,
            imgsz=imgsz,
            device=self.device
        )
        print("Validation completed.")
        return metrics

    def predict(self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4, return_raw: bool = False):
        if not self.predict_model:
            self.load_model(mode='predict')

        if isinstance(image, str):
            input_image = image
        elif isinstance(image, np.ndarray):
            input_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            input_image = image
        else:
            raise ValueError("Unsupported image format.")

        results = self.predict_model.predict(
            source=input_image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device
        )
        
        if return_raw:
            return {
                "raw": results,
                "boxes": results[0].boxes.xyxy,
                "scores": results[0].boxes.conf,
                "classes": results[0].boxes.cls
            }
        else:
            annotated_image = results[0].plot()
            return annotated_image
        
if __name__=="__main__":
    trainer = YOLOTrainer(
        model_path='models/yolov5s.pt', 
        config_path='configs/custom.yaml'
    )
    trainer.train(epochs=100, imgsz=256, batch_size=4, project='results', name='train1')

    metrics = trainer.validate(imgsz=256)
    print(metrics)  # mAP, precision, recall 출력

    results = trainer.infer(
        source='datasets/images/val/', 
        save=True, 
        save_txt=True, 
        project='results', 
        name='my_detection'
    )

    results = trainer.predict(
        source="datasets/images/val",  # 추론할 데이터 경로
        conf=0.25,  # Confidence Threshold
        iou=0.45,   # IoU Threshold
        save=True,
    )