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

