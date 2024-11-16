import torch
import torch.nn as nn

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1) for in_channels in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        last_inner = self.lateral_convs[-1](features[-1])
        results = [self.fpn_convs[-1](last_inner)]
        
        for i in range(len(features) - 2, -1, -1):
            inner_top_down = nn.functional.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = self.lateral_convs[i](features[i])
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.fpn_convs[i](last_inner))
        
        return results
