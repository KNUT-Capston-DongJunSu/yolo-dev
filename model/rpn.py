import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

class RPN(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int = 9) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.cls_logits = nn.Conv2d(512, num_anchors * 2, kernel_size=1, stride=1)  # 객체 존재 여부 예측
        self.bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1, stride=1)   # 바운딩 박스 변위 예측

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        objectness = self.cls_logits(x)
        bbox_deltas = self.bbox_pred(x)
        return objectness, bbox_deltas  # 객체 존재 확률과 박스 좌표 변위
