class PANet(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(PANet, self).__init__()
        self.pan_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, padding=1) for in_channels in in_channels_list
        ])

    def forward(self, features):
        results = [features[0]]
        for i in range(1, len(features)):
            upsampled = nn.functional.interpolate(results[-1], scale_factor=2, mode="nearest")
            results.append(self.pan_convs[i](features[i] + upsampled))
        return results
