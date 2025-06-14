import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ultralytics import YOLO

def val(model_path, data_yaml):
    # 모델 경로 설정
    model = YOLO(model_path)

    # 모델1 성능 평가
    metrics = model.val(data=data_yaml, split="val")
    print("\n[모델 성능]")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

if __name__=='__main__':
    val('results/medium.pt','config/custom.yaml')

    

