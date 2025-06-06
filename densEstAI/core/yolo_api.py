import torch, cv2
from ultralytics import YOLO
from densEstAI.core.preprocessing import PreProcessor

class YoloAPI:
    def __init__(self, model_path):    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)

    def train_yolo(self, config_path, epochs=100, imgsz=640, batch=16, project='results', name=None, lr0=0.01, optimizer='SGD', **kwargs):
        return self.model.train(
            data=config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            lr0=lr0,
            optimizer=optimizer,
            device=self.device,
            **kwargs
        )

    def smart_predict_yolo(self, frame, stream=False, imgsz=640, conf=0.5, iou=0.7, max_det=300, **kwargs):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = self.model.predict(
            source=frame,
            stream=stream,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=self.device,
            max_det=max_det,
            **kwargs
            )[0]
        return self.process_predicted_results(result)

    def predict_yolo(self, frame, stream=False, imgsz=640, conf=0.5, iou=0.7, max_det=300, **kwargs):
        self.preprocess = PreProcessor(self.model, imgsz)
        frame = self.preprocess.preprocess(frame)

        result = self.model(
            source=frame,
            stream=stream,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=self.device,
            max_det=max_det,
            **kwargs
            )[0]
        return self.process_predicted_results(result)
    
    @staticmethod
<<<<<<< HEAD
    def process_predicted_results(result):
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        classes = result.boxes.cls
=======
    def process_predicted_results(result, target_cls=0):
        boxes = result.boxes
        filtered_boxes = boxes[boxes.cls == target_cls]
        
        boxes = filtered_boxes.xyxy
        confidences = filtered_boxes.conf
        class_ids = filtered_boxes.cls
>>>>>>> 07789bdcb9077836fde309233581de72f38808fb

        if isinstance(boxes, torch.Tensor): boxes = boxes.tolist()
        if isinstance(confidences, torch.Tensor): confidences = confidences.tolist()
        if isinstance(classes, torch.Tensor): classes = classes.tolist()
        
<<<<<<< HEAD
        data_list = [box + [conf, cls] for box, conf, cls in zip(boxes, confidences, classes)]
=======
        data_list = [box + [conf, cls] for box, conf, cls in zip(boxes, confidences, class_ids)]
>>>>>>> 07789bdcb9077836fde309233581de72f38808fb
        results = torch.tensor(data_list, dtype=torch.float32)
        return results

    
        