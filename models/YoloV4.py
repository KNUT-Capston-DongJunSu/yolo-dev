import torch
import torch.nn as nn
from models.CSPnet import CSPBlock
from models.PAnet import PANet
from models.FPN import FPN
from models.PostProcessor import YOLOPostProcessor

class YOLOv4Backbone(nn.Module):
    def __init__(self):
        super(YOLOv4Backbone, self).__init__()
        self.csp1 = CSPBlock(3, 64, 1)
        self.csp2 = CSPBlock(64, 128, 2)
        self.csp3 = CSPBlock(128, 256, 8)
        self.csp4 = CSPBlock(256, 512, 8)
        self.csp5 = CSPBlock(512, 1024, 4)
    
    def forward(self, x):
        out1 = self.csp1(x)
        out2 = self.csp2(out1)
        out3 = self.csp3(out2)  # 작은 해상도
        out4 = self.csp4(out3)  # 중간 해상도
        out5 = self.csp5(out4)  # 가장 작은 해상도
        return out3, out4, out5  # FPN에 전달

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
    def __init__(self, num_classes, input_dim=416, conf_threshold=0.5, iou_threshold=0.4):
        super(YOLOv4, self).__init__()
        self.backbone = YOLOv4Backbone()
        self.head = YOLOv4Head(num_classes)
        self.post_processor = YOLOPostProcessor(conf_threshold=conf_threshold, iou_threshold=iou_threshold, input_dim=input_dim)

    def forward(self, x, original_img_shape):
        features = self.backbone(x)
        outputs = self.head(features)

        # YOLO의 원본 예측 결과
        raw_predictions = torch.cat(outputs, dim=1)  # 예측 결과를 하나로 결합 (각 스케일)
        
        # 후처리 적용 (필터링 + NMS)
        final_boxes = self.post_processor.process_predictions(raw_predictions, original_img_shape)
        
        return final_boxes  # 후처리된 최종 예측 결과 반환