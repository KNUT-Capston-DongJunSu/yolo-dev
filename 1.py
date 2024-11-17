import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from typing import Optional, Tuple, Dict

class CustomDetectionDataset(Dataset):
    def __init__(self, data_set_path: str, odgt_file: str, transforms: Optional[transforms.Compose] = None) -> None:
        self.data_set_path = data_set_path
        self.transforms = transforms
        self.image_files = []
        self.annotations = []

        # odgt 파일 읽기
        with open(odgt_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.image_files.append(f"{entry['ID']}.jpg")
                self.annotations.append(entry['gtboxes'])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 이미지 파일 불러오기
        image_path = os.path.join(self.data_set_path, self.image_files[index])
        image = Image.open(image_path).convert("RGB")
        
        # 해당 이미지의 어노테이션 가져오기
        annotation = self.annotations[index]
        
        # 여러 바운딩 박스와 레이블 정보 가져오기
        bboxes = []
        labels = []
        for box in annotation:
            bboxes.append(box['hbox'])  # 바운딩 박스 좌표
            labels.append(box['tag'])  # 클래스 레이블
        
        # 클래스 레이블 매핑 (예: "person" -> 1)
        label_map = {'person': 1}  # 레이블 매핑 정의
        labels = [label_map.get(label, 0) for label in labels]  # 기본값: 0 (알 수 없는 레이블)

        # 바운딩 박스와 레이블을 텐서로 변환
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # 변환 적용
        if self.transforms:
            image = self.transforms(image)
        
        # 타겟 딕셔너리 반환 (unsqueeze 제거)
        target = {"boxes": bboxes, "labels": labels}
        
        return image, target

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
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
        self.model = YOLOv4(num_classes=2, input_dim=416).to(self.device)
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
            pred_classes = pred_tensor[:, 5]  # 단일 클래스 점수
            pred_boxes = pred_tensor[:, :4]  # 바운딩 박스
            pred_objectness = pred_tensor[:, 4]  # 신뢰도 예측

            # 타겟 처리
            target_labels = target['labels']  # 타겟 클래스
            target_boxes = target['boxes']  # 타겟 박스
            target_objectness = torch.ones_like(pred_objectness)  # 타겟 신뢰도 (객체 존재 여부)

            # 손실 계산
            cls_loss = F.binary_cross_entropy_with_logits(pred_classes, target_labels.float())
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
    
import torch
import torch.nn as nn

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPBlock, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.shortcut_branch = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            ) for _ in range(num_blocks)]
        )
        
        self.concat_conv = nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        main = self.main_branch(x)
        shortcut = self.shortcut_branch(x)
        main = self.blocks(main)
        x = torch.cat([main, shortcut], dim=1)
        return self.concat_conv(x)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        # Lateral convolutions for each level
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        # FPN convolutions for each level
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])
        # Channel alignment for top-down pathway
        self.channel_align = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=1) for _ in in_channels_list
        ])

    def forward(self, features):
        print(f"FPN input features: {[f.shape for f in features]}")

        # Process the last feature map (smallest resolution)
        last_inner = self.lateral_convs[-1](features[-1])
        print(f"Last inner shape (initial): {last_inner.shape}")

        results = [self.fpn_convs[-1](last_inner)]

        for i in range(len(features) - 2, -1, -1):
            # Upsample and align the top-down feature map
            inner_top_down = F.interpolate(last_inner, size=features[i].shape[-2:], mode="nearest")
            inner_top_down = self.channel_align[i](inner_top_down)  # Align channels

            # Process the lateral feature map
            inner_lateral = self.lateral_convs[i](features[i])
            print(f"Inner Top-Down shape: {inner_top_down.shape}, Inner Lateral shape: {inner_lateral.shape}")

            # Combine lateral and top-down features
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.fpn_convs[i](last_inner))

        print(f"FPN output features: {[r.shape for r in results]}")
        return results


import torch
import torch.nn as nn
from models.CSPnet import CSPBlock
from models.PAnet import PANet
from models.FPN import FPN
from models.PostProcessor import YOLOPostProcessor

class YOLOv4Backbone(nn.Module):
    def __init__(self):
        super(YOLOv4Backbone, self).__init__()
        self.csp1 = CSPBlock(3, 32, 1)       # 채널 크기 감소
        self.csp2 = CSPBlock(32, 64, 1)      # 채널 크기 감소, 블록 수 감소
        self.csp3 = CSPBlock(64, 128, 2)     # 채널 크기 감소, 블록 수 감소
        self.csp4 = CSPBlock(128, 256, 2)    # 채널 크기 감소, 블록 수 감소
        self.csp5 = CSPBlock(256, 512, 1)    # 채널 크기 감소, 블록 수 감소

    
    def forward(self, x):
        out1 = self.csp1(x)
        out2 = self.csp2(out1)
        out3 = self.csp3(out2)  # 작은 해상도
        out4 = self.csp4(out3)  # 중간 해상도
        out5 = self.csp5(out4)  # 가장 작은 해상도
        print(f"out3: {out3.shape}, out4: {out4.shape}, out5: {out5.shape}")
        return out3, out4, out5  # FPN에 전달

class YOLOv4Head(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv4Head, self).__init__()
        self.fpn = FPN([128, 256, 512], 128)  # FPN 경량화
        self.pan = PANet([128, 128, 128], 128)  # PANet 경량화
        self.num_classes = num_classes
        self.yolo_layers = nn.ModuleList([
            nn.Conv2d(128, 3 * (num_classes + 5), 1) for _ in range(3)  # 3개의 스케일별 YOLO 레이어
        ])

    def forward(self, features):
        # FPN
        fpn_features = self.fpn(features)
        print(f"FPN outputs: {[f.shape for f in fpn_features]}")

        # PANet
        pan_features = self.pan(fpn_features)
        outputs = []

        for layer, p in zip(self.yolo_layers, pan_features):
            out = layer(p)  # [batch, 3 * (num_classes + 5), H, W]
            batch_size, _, h, w = out.shape
            # YOLO 형식으로 reshape
            out = out.view(batch_size, 3, self.num_classes + 5, h, w).permute(0, 1, 3, 4, 2)
            # 시그모이드 활성화
            out[..., 4] = torch.sigmoid(out[..., 4])  # Confidence score
            out[..., 5:] = torch.sigmoid(out[..., 5:])  # Class probabilities
            outputs.append(out)

        return outputs  # [scale1, scale2, scale3]


class YOLOv4(nn.Module):
    def __init__(self, num_classes, input_dim=416, conf_threshold=0.5, iou_threshold=0.4):
        super(YOLOv4, self).__init__()
        self.backbone = YOLOv4Backbone()
        self.head = YOLOv4Head(num_classes)
        self.post_processor = YOLOPostProcessor(conf_threshold=conf_threshold, iou_threshold=iou_threshold, input_dim=input_dim)

    def forward(self, x, original_img_shape):
        print("Starting Backbone...")
        features = self.backbone(x)
        print(f"Backbone outputs: {[f.shape for f in features]}")

        print("Starting Head...")
        outputs = self.head(features)
        print(f"Head outputs: {[o.shape for o in outputs]}")

        print("Concatenating outputs...")
        raw_predictions = torch.cat(outputs, dim=1)
        print(f"Raw predictions shape: {raw_predictions.shape}")

        print("Post-processing...")
        final_boxes = self.post_processor.process_predictions(raw_predictions, original_img_shape)
        print("Forward pass completed.")
        return final_boxes


import torch
from torchvision.ops import nms

class YOLOPostProcessor:
    def __init__(self, conf_threshold=0.3, iou_threshold=0.4, input_dim=256):
        """
        YOLO 모델의 추론 결과를 처리하는 PostProcessor.
        
        Args:
            conf_threshold (float): 신뢰도 점수 임계값 (default=0.3)
            iou_threshold (float): NMS에 사용할 IoU 임계값 (default=0.4)
            input_dim (int): 모델 입력 크기 (default=256)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_dim = input_dim

    def process_predictions(self, predictions, original_img_shape):
        """
        YOLO 모델의 원본 예측 결과를 후처리하여 최종 바운딩 박스 생성.

        Args:
            predictions (torch.Tensor): YOLO 모델의 출력. Shape: [batch_size, num_predictions, num_classes + 5]
            original_img_shape (tuple): 원본 이미지 크기 (height, width)

        Returns:
            list: 최종 바운딩 박스 리스트. 각 박스는 [x, y, w, h, confidence, class_id, class_score].
        """
        # 디버깅: original_img_shape 출력
        #print(f"Processing predictions with original image shape: {original_img_shape}")

        batch_boxes = []
        for batch_idx, pred in enumerate(predictions):
            # print(f"Processing batch {batch_idx}, raw predictions shape: {pred.shape}")

            # Filter boxes
            filtered_boxes = self.filter_boxes(pred)
            # print(f"Batch {batch_idx} - Filtered boxes count: {len(filtered_boxes)}")

            if len(filtered_boxes) == 0:
                # print(f"No boxes left after filtering for batch {batch_idx}. Skipping...")
                batch_boxes.append([])
                continue

            # Rescale boxes
            rescaled_boxes = self.rescale_boxes(filtered_boxes, original_img_shape)
            # print(f"Batch {batch_idx} - Rescaled boxes count: {len(rescaled_boxes)}")

            if len(rescaled_boxes) == 0:
                # print(f"No boxes left after rescaling for batch {batch_idx}. Skipping...")
                batch_boxes.append([])
                continue

            # Non-Max Suppression
            nms_boxes = self.non_max_suppression(rescaled_boxes)
            # print(f"Batch {batch_idx} - NMS boxes count: {len(nms_boxes)}")

            batch_boxes.append(nms_boxes)
        return batch_boxes

    def filter_boxes(self, predictions, max_boxes=3000):
        """
        신뢰도 점수가 일정 기준 이상인 바운딩 박스를 필터링.
        
        Args:
            predictions (torch.Tensor): 예측 결과.
            max_boxes (int): 유지할 최대 박스 수.
        
        Returns:
            list: 필터링된 바운딩 박스.
        """
        mask = predictions[..., 4] > self.conf_threshold
        # print(f"Confidence scores: min={predictions[..., 4].min()}, max={predictions[..., 4].max()}, mean={predictions[..., 4].mean()}")
        # print(f"Mask shape: {mask.shape}, Passed boxes: {mask.sum().item()}")

        filtered_preds = predictions[mask]

        # 상위 max_boxes 제한
        if len(filtered_preds) > max_boxes:
            _, indices = filtered_preds[..., 4].topk(max_boxes)
            filtered_preds = filtered_preds[indices]

        boxes = []
        for pred in filtered_preds:
            x, y, w, h = pred[:4]
            conf = pred[4]
            class_scores = pred[5:]
            class_id = class_scores.argmax().item()
            class_score = class_scores[class_id].item()
            boxes.append([x, y, w, h, conf, class_id, class_score])
        return boxes

    def rescale_boxes(self, boxes, original_img_shape):
        """
        모델 입력 크기에서 원본 이미지 크기로 바운딩 박스 크기를 조정.
        
        Args:
            boxes (list): 바운딩 박스 리스트.
            original_img_shape (tuple): 원본 이미지 크기 (height, width).
        
        Returns:
            list: 원본 이미지 크기에 맞게 조정된 바운딩 박스 리스트.
        """
        orig_h, orig_w = original_img_shape
        scale_factor = min(self.input_dim / orig_w, self.input_dim / orig_h)

        scaled_boxes = []
        for box in boxes:
            x = (box[0] - (self.input_dim - scale_factor * orig_w) / 2) / scale_factor
            y = (box[1] - (self.input_dim - scale_factor * orig_h) / 2) / scale_factor
            w = box[2] / scale_factor
            h = box[3] / scale_factor

            # 좌표 범위 제한 (0 이상)
            x = max(0, x)
            y = max(0, y)
            scaled_boxes.append([x, y, w, h, box[4], box[5], box[6]])
        return scaled_boxes

    def non_max_suppression(self, boxes):
        """
        Non-Maximum Suppression (NMS) 알고리즘을 통해 중복 바운딩 박스 제거.
        
        Args:
            boxes (list): 바운딩 박스 리스트.
        
        Returns:
            list: NMS를 통과한 바운딩 박스 리스트.
        """
        if len(boxes) == 0:
            return []

        boxes_tensor = torch.tensor(boxes)
        xywh_boxes = boxes_tensor[:, :4]
        scores = boxes_tensor[:, 4]
        indices = nms(xywh_boxes, scores, self.iou_threshold)
        return boxes_tensor[indices].tolist()

    def calculate_iou(self, box1, box2):
        """
        두 바운딩 박스 간 IoU(Intersection over Union) 계산.
        
        Args:
            box1 (list): 첫 번째 박스 [x, y, w, h, ...].
            box2 (list): 두 번째 박스 [x, y, w, h, ...].
        
        Returns:
            float: IoU 값.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]

        return inter_area / (box1_area + box2_area - inter_area + 1e-6)



if __name__ == "__main__":
    main = Main(batch_size=2, num_epochs=10)  # 배치 크기와 에포크 설정
    main.train()

