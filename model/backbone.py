import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.utils import make_layers
from typing import Union, List

class Backbone(nn.Module):
    
    CONFIGS: dict[str, List[Union[int, str]]] = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }

    def __init__(self, config_name: str = 'vgg16', batch_norm: bool = False) -> None:
        super().__init__()
        cfg = self.CONFIGS.get(config_name, self.CONFIGS['vgg16'])
        self.features = make_layers(cfg, batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x  # 특징 맵 반환
