import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from model_YOLO.YoloV4 import YOLOv4  # YOLO 스타일의 모델을 정의
from dataset.customdataset import CustomDetectionDataset
from torchvision import transforms
import numpy as np
from typing import List, Dict

class Main:
    dataset_path: str = 'path/to/images'
    annotation_file: str = 'path/to/annotations.json'
    
    def __init__(self, batch_size: int = 64, num_epochs: int = 1) -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.get_data()

        # YOLO 모델 초기화
        self.model = YOLOv4(num_classes=1, input_dim=416)  # YOLOv4 모델
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_data(self) -> None:
        # 데이터 로더 설정
        data_transforms = transforms.Compose([
            transforms.Resize((416, 416)),  # YOLO 모델에 맞게 입력 크기 조정
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 데이터 증강
            transforms.RandomHorizontalFlip(),
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
                
                # 예측 수행
                predictions = self.model(images)
                
                # 손실 계산
                total_loss = self.compute_loss(predictions, targets)
                total_loss.backward()
                self.optimizer.step()
                
                cumulative_loss += total_loss.item()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {cumulative_loss / len(self.train_loader)}")
            
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
                images = images[0]
                targets = targets[0]
                
                # 예측 수행
                predictions = self.model(images)
                
                # 손실 계산
                total_loss += self.compute_loss(predictions, targets).item()
                
                # 후처리로 NMS 적용 후 결과 얻기
                post_processor = YOLOPostProcessor(conf_threshold=0.5, iou_threshold=0.4, input_dim=416)
                final_boxes = post_processor.process_predictions(predictions, original_img_shape=images.shape[2:])
                
                # IoU 계산
                for pred_box, true_box in zip(final_boxes, targets['boxes']):
                    iou = self.calculate_iou(pred_box, true_box.numpy())
                    ious.append(iou)
        
        # 평균 손실과 평균 IoU 출력
        mean_loss = total_loss / len(self.test_loader)
        mean_iou = np.mean(ious)
        print(f"Validation Loss: {mean_loss}, Mean IoU: {mean_iou}")

    def test(self) -> None:
        self.model.load_state_dict(torch.load('trained_detector.pth'))
        self.model.eval()  # 평가 모드로 설정
        
        # 테스트 루프
        ious: List[float] = []  # IoU 점수를 저장할 리스트
        for images, targets in self.test_loader:
            images = images[0]  # 배치 크기가 1이므로 [0]으로 접근
            targets = targets[0]
            
            with torch.no_grad():
                # 예측 수행
                predictions = self.model(images)
                
                # 후처리로 NMS 적용 후 결과 얻기
                post_processor = YOLOPostProcessor(conf_threshold=0.5, iou_threshold=0.4, input_dim=416)
                final_boxes = post_processor.process_predictions(predictions, original_img_shape=images.shape[2:])
                
                # IoU 계산
                for pred_box, true_box in zip(final_boxes, targets['boxes']):
                    iou = self.calculate_iou(pred_box, true_box.numpy())
                    ious.append(iou)
                    print(f"Predicted Box: {pred_box}, True Box: {true_box}, IoU: {iou}")

        # 평균 IoU 출력
        mean_iou = np.mean(ious)
        print(f"Mean IoU: {mean_iou}")

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        YOLO 손실 계산 함수
        - predictions: 모델의 예측 결과, 예: {'boxes': ..., 'classes': ..., 'objectness': ...}
        - targets: 실제 레이블, 예: {'boxes': ..., 'labels': ..., 'objectness': ...}
        """
        # 클래스 손실
        cls_loss = F.cross_entropy(predictions['classes'], targets['labels'].long())
        
        # 바운딩 박스 손실
        bbox_loss = F.smooth_l1_loss(predictions['boxes'], targets['boxes'])
        
        # 객체성 손실
        objectness_loss = F.binary_cross_entropy_with_logits(predictions['objectness'], targets['objectness'])
        
        total_loss = cls_loss + bbox_loss + objectness_loss
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
    main.train()
    main.test()
