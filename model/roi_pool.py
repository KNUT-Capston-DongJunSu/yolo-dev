import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import List, Tuple, Union

class RoIPool(nn.Module):
    def __init__(self, output_size: Union[int, Tuple[int, int]], spatial_scale: float = 1.0) -> None:
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, feature_map: torch.Tensor, proposals: List[Tuple[float, float, float, float]]) -> torch.Tensor:
        pooled_features = []
        for proposal in proposals:
            x1, y1, x2, y2 = [int(coord * self.spatial_scale) for coord in proposal]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(feature_map.shape[-1], x2)
            y2 = min(feature_map.shape[-2], y2)
            roi = feature_map[..., y1:y2, x1:x2]
            pooled_features.append(F.adaptive_max_pool2d(roi, self.output_size))
        return torch.stack(pooled_features)
