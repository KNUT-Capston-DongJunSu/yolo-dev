import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from densEstAI.core import YoloManager  

if __name__=="__main__":
    model = YoloManager(model_path="yolov8m.pt")
    model.train_yolo(
        config_path="config/custom.yaml",
        epochs=20,              # 추가로 10 epoch만 학습
        imgsz=512,
        batch=8,
        resume=False,           # "이어하기"가 아니라 "새로운 학습"으로 처리
        project="results",
        name="train3",  # 결과 폴더 다르게 해서 구분
        patience=5              # 5 epoch 동안 개선 없으면 자동 종료
    )
    
    '''model.train_yolo(
        config_path="config/custom.yaml",
        resume=True
        )'''