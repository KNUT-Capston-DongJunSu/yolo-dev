import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from model.simple_object_detector import SimpleObjectDetector
from dataset.customdataset import CustomDetectionDataset
from torchvision import transforms
import numpy as np
from typing import List, Tuple, Dict

class Main:
    dataset_path: str = 'path/to/images'
    annotation_file: str = 'path/to/annotations.json'
    
    def __init__(self, batch_size: int = 64, num_epochs: int = 1) -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.get_data()

        # 모델 인스턴스화 및 손실 함수, 옵티마이저 정의
        self.model = SimpleObjectDetector(num_classes=3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_data(self) -> None:
        # 데이터 로더 설정
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = CustomDetectionDataset(
            data_set_path=Main.dataset_path,
            annotation_file=Main.annotation_file,
            transforms=data_transforms
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
        )

        test_dataset = CustomDetectionDataset(
            data_set_path=Main.dataset_path,
            annotation_file=Main.annotation_file,
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
                self.optimizer.zero_grad()
                
                # 모델 예측 수행
                proposals = self.generate_proposals(images)
                cls_scores, bbox_preds, objectness, bbox_deltas = self.model(images, proposals)
                
                # 총 손실 계산 및 역전파
                total_loss = self.compute_loss(cls_scores, bbox_preds, objectness, bbox_deltas, targets)
                total_loss.backward()
                self.optimizer.step()
                
                cumulative_loss += total_loss.item()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {cumulative_loss / len(self.train_loader)}")

        # 모델 저장
        torch.save(self.model.state_dict(), 'trained_detector.pth')

    def test(self) -> None:
        self.model.load_state_dict(torch.load('trained_detector.pth'))
        self.model.eval()  # 평가 모드로 설정
        
        # 테스트 루프
        ious: List[float] = []  # IoU 점수를 저장할 리스트
        for images, targets in self.test_loader:
            images = images[0]  # 배치 크기가 1이므로 [0]으로 접근
            targets = targets[0]
            
            with torch.no_grad():
                # 모델 예측 수행
                proposals = self.generate_proposals(images)
                cls_scores, bbox_preds, objectness, bbox_deltas = self.model(images, proposals)
                
                # 예측된 바운딩 박스와 정답 바운딩 박스의 IoU 계산
                pred_boxes = bbox_preds.squeeze().numpy()
                true_boxes = targets['boxes'].squeeze().numpy()
                
                # 각 예측과 정답에 대해 IoU 계산
                for pred_box, true_box in zip(pred_boxes, true_boxes):
                    iou = self.calculate_iou(pred_box, true_box)
                    ious.append(iou)
                    print(f"Predicted Box: {pred_box}, True Box: {true_box}, IoU: {iou}")

        # 평균 IoU 출력
        mean_iou = np.mean(ious)
        print(f"Mean IoU: {mean_iou}")

    def generate_proposals(self, images: torch.Tensor) -> torch.Tensor:
        # 임시 제안 생성
        batch_size = images.size(0)
        proposals = [torch.tensor([[50, 30, 150, 120], [30, 40, 100, 180]], dtype=torch.float32) for _ in range(batch_size)]
        return torch.stack(proposals)

    def compute_loss(self, cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor, objectness: torch.Tensor,
        bbox_deltas: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 손실 계산 함수 예시
        classification_loss = F.cross_entropy(cls_scores, targets['labels'].long())
        bbox_loss = F.smooth_l1_loss(bbox_preds, targets['boxes'])
        objectness_loss = F.binary_cross_entropy_with_logits(objectness, targets['objectness'])
        rpn_bbox_loss = F.smooth_l1_loss(bbox_deltas, targets['bbox_deltas'])

        total_loss = classification_loss + bbox_loss + objectness_loss + rpn_bbox_loss
        return total_loss

    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        # IoU 계산 함수
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou

if __name__ == "__main__":
    main = Main()
