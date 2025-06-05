import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from densEstAI import YoloAPI

if __name__=="__main__":
    model = YoloAPI(model_path="results/train152/weights/last.pt")
    model.train_yolo(
        config_path="config/custom.yaml",
        epochs=7,
        imgsz=640,
        batch=12,
        resume=False, 
        project="results",
        name="train15"
        )