import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from models.YoloV4 import YOLOv4  # YOLO 스타일의 모델을 정의
from data.customdataset import CustomDetectionDataset
from models.PostProcessor import YOLOPostProcessor
from torchvision import transforms
import numpy as np
from typing import List, Dict

class Main:
    train_path: str = './dataset/train/Images/'
    val_path: str = './dataset/val/Images/'
    train_file: str = './dataset/train/annotation_train.odgt'
    val_file: str = './dataset/val/annotation_val.odgt'
    
    def __init__(self, batch_size: int = 64, num_epochs: int = 1, device: str = 'cuda') -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        self.get_data()
        # YOLO 모델 초기화
        self.model = YOLOv4(num_classes=1, input_dim=416).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_data(self) -> None:
        # 데이터 로더 설정
        data_transforms = transforms.Compose([
            transforms.Resize((256, 256)),  # YOLO 모델에 맞게 입력 크기 조정
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 데이터 증강
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = CustomDetectionDataset(
            data_set_path=Main.train_path,
            odgt_file=Main.train_file,
            transforms=data_transforms
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
        )

        test_dataset = CustomDetectionDataset(
            data_set_path=Main.val_path,
            odgt_file=Main.val_file,
            transforms=data_transforms
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
        )

    def train(self) -> None:
        # 모델 훈련 루프
        for epoch in range(self.num_epochs):
            self.model.train()
            cumulative_loss = 0
            
            for images, targets in self.train_loader:
                print("Data batch loaded")
                images = torch.stack(images).to(self.device)  # 텐서화 및 GPU로 이동
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()
                
                # 예측 수행
                predictions = self.model(images, original_img_shape=(256, 256))  # 모델 예측
                print("Prediction completed")

                # 손실 계산
                total_loss = self.compute_loss(predictions, targets)
                total_loss.backward()
                self.optimizer.step()
                
                cumulative_loss += total_loss.item()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {cumulative_loss / len(self.train_loader):.4f}")
            
            # 검증 수행
            self.validate()

        # 모델 저장
        torch.save(self.model.state_dict(), 'trained_detector.pth')

    def validate(self) -> None:
        self.model.eval()  # 평가 모드로 설정
        ious: List[float] = []  # IoU 점수를 저장할 리스트
        total_loss = 0

        with torch.no_grad():
            for images, targets in self.test_loader:
                images = torch.stack(images).to(self.device)  # 텐서화 및 GPU로 이동
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # 예측 수행
                predictions = self.model(images, original_img_shape=images.shape[2:])
                
                # 손실 계산
                total_loss += self.compute_loss(predictions, targets).item()
                
                # 후처리로 NMS 적용 후 결과 얻기
                post_processor = YOLOPostProcessor(conf_threshold=0.5, iou_threshold=0.4, input_dim=416)
                final_boxes = post_processor.process_predictions(predictions, original_img_shape=images.shape[2:])
                
                # IoU 계산
                for pred_box, true_box in zip(final_boxes, targets[0]['boxes']):
                    iou = self.calculate_iou(pred_box, true_box.cpu().numpy())
                    ious.append(iou)
        
        # 평균 손실과 평균 IoU 출력
        mean_loss = total_loss / len(self.test_loader)
        mean_iou = np.mean(ious)
        print(f"Validation Loss: {mean_loss:.4f}, Mean IoU: {mean_iou:.4f}")

    def compute_loss(self, predictions: List[List], targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        YOLO 손실 계산 함수
        """
        total_cls_loss = 0.0
        total_bbox_loss = 0.0
        total_objectness_loss = 0.0

        for pred, target in zip(predictions, targets):
            if len(pred) == 0:  # 예측 값이 없을 경우 스킵
                continue

            # 예측 텐서로 변환
            pred_tensor = torch.tensor(pred, dtype=torch.float32)
            pred_classes = pred_tensor[:, 5:]  # 클래스 점수
            pred_boxes = pred_tensor[:, :4]   # 바운딩 박스

            # 타겟 처리
            target_labels = target['labels']  # 타겟 클래스
            target_boxes = target['boxes']  # 타겟 박스
            target_objectness = torch.ones_like(pred_classes[:, 0])  # 타겟 신뢰도 (객체 존재 여부)

            # 타겟 레이블을 one-hot 인코딩 (예: 클래스 0이면 [1, 0], 클래스 1이면 [0, 1])
            target_labels_one_hot = torch.zeros_like(pred_classes)
            for i, label in enumerate(target_labels):
                target_labels_one_hot[i, label] = 1

            # 손실 계산
            cls_loss = F.binary_cross_entropy_with_logits(pred_classes, target_labels_one_hot.float())
            bbox_loss = F.smooth_l1_loss(pred_boxes, target_boxes)
            objectness_loss = F.binary_cross_entropy_with_logits(pred_objectness, target_objectness)

            # 배치별 손실 누적
            total_cls_loss += cls_loss
            total_bbox_loss += bbox_loss
            total_objectness_loss += objectness_loss

        total_loss = total_cls_loss + total_bbox_loss + total_objectness_loss

        # 디버깅 출력
        print(f"Total Loss: {total_loss}, CLS Loss: {total_cls_loss}, BBOX Loss: {total_bbox_loss}, OBJ Loss: {total_objectness_loss}")

        return total_loss






    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        IoU 계산 함수
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou

if __name__ == "__main__":
    main = Main(batch_size=2, num_epochs=10)  # 배치 크기와 에포크 설정
    main.train()
