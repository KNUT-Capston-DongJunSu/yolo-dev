import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import List, Tuple

class SimpleObjectDetector(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.rpn = RPN(in_channels=256)
        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=0.5)  # 예시로 feature_map의 크기 조정
        self.head = Head(in_features=256 * 7 * 7, num_classes=num_classes)

    def forward(self, x: torch.Tensor, 
                proposals: List[Tuple[float, float, float, float]]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_map = self.backbone(x)
        objectness, bbox_deltas = self.rpn(feature_map)

        # RoI Pooling을 통해 프로포절 기반 특징 추출
        roi_features = self.roi_pool(feature_map, proposals)
        cls_scores, bbox_preds = self.head(roi_features.view(roi_features.size(0), -1))

        return cls_scores, bbox_preds, objectness, bbox_deltas
