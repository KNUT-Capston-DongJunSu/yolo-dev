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
