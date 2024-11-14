import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Dict

# 손실 함수 예시
def compute_loss(cls_scores: torch.Tensor, bbox_preds: torch.Tensor, 
    objectness: torch.Tensor, bbox_deltas: torch.Tensor, 
    targets: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
    # 1. 분류 손실 (클래스 예측 손실)
    classification_loss = F.cross_entropy(cls_scores, torch.stack(targets['labels']).squeeze(1))
    # 2. 최종 바운딩 박스 회귀 손실
    bbox_loss = F.smooth_l1_loss(bbox_preds, torch.stack(targets['boxes']).squeeze(1))
    # 3. 객체 존재 확률 손실 (RPN의 객체 여부 손실)
    objectness_loss = F.binary_cross_entropy_with_logits(objectness, torch.stack(targets['objectness']))
    # 4. RPN 바운딩 박스 변위 손실
    rpn_bbox_loss = F.smooth_l1_loss(bbox_deltas, torch.stack(targets['bbox_deltas']))\
    # 전체 손실 계산
    total_loss = classification_loss + bbox_loss + objectness_loss + rpn_bbox_loss
    return total_loss

# IoU 계산 함수
def calculate_iou(box1: List[float], box2: List[float]) -> float:
    # box는 [x1, y1, x2, y2] 형식이어야 합니다.
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

# 제안된 영역 생성 함수
def generate_proposals(images: torch.Tensor) -> List[torch.Tensor]:
    batch_size = images.size(0)
    proposals = [
        torch.tensor([[50, 30, 150, 120], [30, 40, 100, 180]], dtype=torch.float32) 
        for _ in range(batch_size)
    ]
    return proposals

# 모델 레이어 생성 함수
def make_layers(cfg: List, batch_norm: bool = False) -> nn.Sequential:
    """Helper function to create layers from configuration."""
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if batch_norm:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ]
            in_channels = v
    return nn.Sequential(*layers)
