class YOLOv4Backbone(nn.Module):
    def __init__(self):
        super(YOLOv4Backbone, self).__init__()
        self.csp1 = CSPBlock(3, 64, 1)  # 첫 번째 CSP 블록
        self.csp2 = CSPBlock(64, 128, 2)
        self.csp3 = CSPBlock(128, 256, 8)
        self.csp4 = CSPBlock(256, 512, 8)
        self.csp5 = CSPBlock(512, 1024, 4)

    def forward(self, x):
        out1 = self.csp1(x)
        out2 = self.csp2(out1)
        out3 = self.csp3(out2)
        out4 = self.csp4(out3)
        out5 = self.csp5(out4)
        return out3, out4, out5  # 다양한 크기의 피처 맵을 반환


class YOLOv4Head(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv4Head, self).__init__()
        self.fpn = FPN([256, 512, 1024], 256)
        self.pan = PANet([256, 256, 256], 256)
        self.yolo_layers = nn.ModuleList([
            nn.Conv2d(256, 3 * (num_classes + 5), 1) for _ in range(3)  # 3개의 스케일별 YOLO 레이어
        ])

    def forward(self, features):
        fpn_features = self.fpn(features)
        pan_features = self.pan(fpn_features)
        outputs = [layer(p) for layer, p in zip(self.yolo_layers, pan_features)]
        return outputs


class YOLOv4(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv4, self).__init__()
        self.backbone = YOLOv4Backbone()
        self.head = YOLOv4Head(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs
