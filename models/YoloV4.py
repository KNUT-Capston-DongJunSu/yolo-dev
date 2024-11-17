import torch
import torch.nn as nn
from models.CSPnet import CSPBlock
from models.PAnet import PANet
from models.FPN import FPN
from models.PostProcessor import YOLOPostProcessor

class YOLOv4Backbone(nn.Module):
    def __init__(self):
        super(YOLOv4Backbone, self).__init__()
        self.csp1 = CSPBlock(3, 32, 1)       # 채널 크기 감소
        self.csp2 = CSPBlock(32, 64, 1)      # 채널 크기 감소, 블록 수 감소
        self.csp3 = CSPBlock(64, 128, 2)     # 채널 크기 감소, 블록 수 감소
        self.csp4 = CSPBlock(128, 256, 2)    # 채널 크기 감소, 블록 수 감소
        self.csp5 = CSPBlock(256, 512, 1)    # 채널 크기 감소, 블록 수 감소

    
    def forward(self, x):
        out1 = self.csp1(x)
        out2 = self.csp2(out1)
        out3 = self.csp3(out2)  # 작은 해상도
        out4 = self.csp4(out3)  # 중간 해상도
        out5 = self.csp5(out4)  # 가장 작은 해상도
        print(f"out3: {out3.shape}, out4: {out4.shape}, out5: {out5.shape}")
        return out3, out4, out5  # FPN에 전달

class YOLOv4Head(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv4Head, self).__init__()
        self.fpn = FPN([128, 256, 512], 128)  # FPN 경량화
        self.pan = PANet([128, 128, 128], 128)  # PANet 경량화
        self.num_classes = num_classes
        self.yolo_layers = nn.ModuleList([
            nn.Conv2d(128, 3 * (num_classes + 5), 1) for _ in range(3)  # 3개의 스케일별 YOLO 레이어
        ])

    def forward(self, features):
        # FPN
        fpn_features = self.fpn(features)
        print(f"FPN outputs: {[f.shape for f in fpn_features]}")

        # PANet
        pan_features = self.pan(fpn_features)
        outputs = []

        for layer, p in zip(self.yolo_layers, pan_features):
            out = layer(p)  # [batch, 3 * (num_classes + 5), H, W]
            batch_size, _, h, w = out.shape
            # YOLO 형식으로 reshape
            out = out.view(batch_size, 3, self.num_classes + 5, h, w).permute(0, 1, 3, 4, 2)
            # 시그모이드 활성화
            out[..., 4] = torch.sigmoid(out[..., 4])  # Confidence score
            out[..., 5:] = torch.sigmoid(out[..., 5:])  # Class probabilities
            outputs.append(out)

        return outputs  # [scale1, scale2, scale3]


class YOLOv4(nn.Module):
    def __init__(self, num_classes, input_dim=416, conf_threshold=0.5, iou_threshold=0.4):
        super(YOLOv4, self).__init__()
        self.backbone = YOLOv4Backbone()
        self.head = YOLOv4Head(num_classes)
        self.post_processor = YOLOPostProcessor(conf_threshold=conf_threshold, iou_threshold=iou_threshold, input_dim=input_dim)

    def forward(self, x, original_img_shape):
        print("Starting Backbone...")
        features = self.backbone(x)
        print(f"Backbone outputs: {[f.shape for f in features]}")

        print("Starting Head...")
        outputs = self.head(features)
        print(f"Head outputs: {[o.shape for o in outputs]}")

        print("Concatenating outputs...")
        raw_predictions = torch.cat(outputs, dim=1)
        print(f"Raw predictions shape: {raw_predictions.shape}")

        print("Post-processing...")
        final_boxes = self.post_processor.process_predictions(raw_predictions, original_img_shape)
        print("Forward pass completed.")
        return final_boxes