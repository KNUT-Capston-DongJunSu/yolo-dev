import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from densEstAI import YoloAPI

if __name__=="__main__":
<<<<<<< HEAD
    model = YoloAPI(model_path="results/train15/weights/last.pt")
    model.train_yolo(
        config_path="config/custom.yaml",
        epochs=17,
=======
    model = YoloAPI(model_path="results/train152/weights/last.pt")
    model.train_yolo(
        config_path="config/custom.yaml",
        epochs=7,
>>>>>>> 07789bdcb9077836fde309233581de72f38808fb
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