from densEstAI import train_yolo

if __name__=="__main__":
    train_yolo(
        model_path="yolov8s.pt",
        config_path="configs/custom.yaml",
        epochs=5,
        imgsz=640,
        batch=12,
        resume=False, 
        project="results",
        name="train15"
        )