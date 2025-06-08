from ultralytics import YOLO

if __name__=='__main__':
    # 모델 경로 설정
    model_5 = YOLO("results/train/weights/best.pt")
    model_20 = YOLO("results/train2/weights/best.pt")  # 20epoch 모델
    model_27 = YOLO("results/train22/weights/best.pt")   # 현재 7epoch 모델

    # 평가할 데이터셋 경로 설정 (val.yaml 또는 val 이미지 폴더)
    data_yaml = "config/custom.yaml"  # 반드시 val 항목 정의되어 있어야 함

    # 모델1 성능 평가
    metrics_5 = model_5.val(data=data_yaml, split="val")
    print("\n[5epoch 모델 성능]")
    print(f"mAP50: {metrics_5.box.map50:.4f}")
    print(f"mAP50-95: {metrics_5.box.map:.4f}")
    print(f"Precision: {metrics_5.box.mp:.4f}")
    print(f"Recall: {metrics_5.box.mr:.4f}")

    # 모델2 성능 평가
    metrics_20 = model_20.val(data=data_yaml, split="val")
    print("\n[20epoch 모델 성능]")
    print(f"mAP50: {metrics_20.box.map50:.4f}")
    print(f"mAP50-95: {metrics_20.box.map:.4f}")
    print(f"Precision: {metrics_20.box.mp:.4f}")
    print(f"Recall: {metrics_20.box.mr:.4f}")

    metrics_27 = model_27.val(data=data_yaml, split="val")
    print("\n[27epoch 모델 성능]")
    print(f"mAP50: {metrics_27.box.map50:.4f}")
    print(f"mAP50-95: {metrics_27.box.map:.4f}")
    print(f"Precision: {metrics_27.box.mp:.4f}")
    print(f"Recall: {metrics_27.box.mr:.4f}")

'''
[5epoch 모델 성능]
mAP50: 0.8977
mAP50-95: 0.6109
Precision: 0.8953
Recall: 0.8758

[20epoch 모델 성능]
mAP50: 0.9182
mAP50-95: 0.6471
Precision: 0.9095
Recall: 0.8832

[27epoch 모델 성능]
mAP50: 0.9137
mAP50-95: 0.6360
Precision: 0.9072
Recall: 0.8729
'''