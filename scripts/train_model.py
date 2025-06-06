from densEstAI.utils import move_to_root_path
move_to_root_path()
from densEstAI.yolo import YoloManager 

if __name__=="__main__":
    model = YoloManager(model_path="results/train15/weights/last.pt")
    model.train_yolo(
        config_path="config/custom.yaml",
        epochs=17,
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