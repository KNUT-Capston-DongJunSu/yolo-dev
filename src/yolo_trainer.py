import torch
from ultralytics import YOLO

class YOLOTrainer:
    def __init__(self, model_path: str = None, config_path: str = None, weight_path: str = None,  device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.weight_path = weight_path
        self.config_path = config_path
        self.train_model = None  # 훈련에 사용될 모델
        self.predict_model = None  # 예측에 사용될 모델
        self.frame_idx = 0
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
        if not self.train_model:
            self.load_model(mode='train')
        self.train_model.train(
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
        if not self.train_model:
            self.load_model(mode='train')
        metrics = self.train_model.val(
            data=self.config_path,
            imgsz=imgsz,
            device=self.device
        )
        print("Validation completed.")
        return metrics
    
    def predict(self, frame, conf_threshold: float = 0.5, iou_threshold: float = 0.4, 
               stream: bool = False, half: bool = False, save: bool = True, project: str = None, name: str = None):
        if not self.predict_model:
            self.load_model(mode='predict')

        # Perform prediction on the current frame
        result = list(
            self.predict_model.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                save=save,
                project=project,
                name=name,
                device=self.device,
                half=half,
            )
        )
        
        # Process raw results for the current frame
        boxes = result[0].boxes.xyxy.tolist() if isinstance(result[0].boxes.xyxy, torch.Tensor) else result[0].boxes.xyxy
        scores = result[0].boxes.conf.tolist() if isinstance(result[0].boxes.conf, torch.Tensor) else result[0].boxes.conf
        classes = result[0].boxes.cls.tolist() if isinstance(result[0].boxes.cls, torch.Tensor) else result[0].boxes.cls

        # Generate plot for the current frame
        plot = result[0].plot()
        self.frame_idx += 1
        
        return {"frame_index": self.frame_idx,
                "prediction": {
                    "boxes": boxes,
                    "scores": scores,
                    "classes": classes
                },
                "plot": plot}

if __name__=="__main__":
    trainer = YOLOTrainer(
        model_path='models/yolov5s.pt', 
        config_path='configs/custom.yaml'
    )
    trainer.train(epochs=100, imgsz=256, batch_size=8, project='results', name='train1')

    metrics = trainer.validate(imgsz=256)
    print(metrics)  # mAP, precision, recall 출력

   
    