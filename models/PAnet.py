import torch
import torch.nn as nn
import torch.nn.functional as F

class PANet(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(PANet, self).__init__()
        # PANet convolutions
        self.pan_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) for in_channels in in_channels_list
        ])
        # 채널 동기화를 위한 Conv2D 추가
        self.channel_align = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=1) for _ in in_channels_list
        ])

    def forward(self, features):
        print(f"PANet input features: {[f.shape for f in features]}")

        results = [self.pan_convs[-1](features[-1])]  # 마지막 Feature Map부터 시작
        for i in range(len(features) - 2, -1, -1):
            # 업샘플링
            upsampled = F.interpolate(results[0], size=features[i].shape[-2:], mode="nearest")
            print(f"Upsampled shape: {upsampled.shape}, Feature shape: {features[i].shape}")

            # 채널 동기화
            upsampled = self.channel_align[i](upsampled)
            print(f"Aligned upsampled shape: {upsampled.shape}")

            # 합산 후 PANet convolution 적용
            results.insert(0, self.pan_convs[i](features[i] + upsampled))

        print(f"PANet output features: {[r.shape for r in results]}")
        return results