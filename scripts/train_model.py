import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from densEstAI.yolo import YoloManager 

if __name__=="__main__":
    model = YoloManager(model_path="yolov8s.pt")
    model.train_yolo(
        config_path="config/custom.yaml",
        epochs=20,
        imgsz=640,
        batch=12,
        resume=False, 
        project="results",
        name="train15"
        )
    
    '''model.train_yolo(
        config_path="config/custom.yaml",
        resume=True
        )'''