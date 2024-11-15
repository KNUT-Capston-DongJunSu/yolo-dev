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

class SPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.pool1 = nn.MaxPool2d(5, 1, 2)
        self.pool2 = nn.MaxPool2d(9, 1, 4)
        self.pool3 = nn.MaxPool2d(13, 1, 6)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x, self.pool1(x), self.pool2(x), self.pool3(x)], dim=1)
        return x



class YOLOv4Backbone(nn.Module):
    def __init__(self):
        super(YOLOv4Backbone, self).__init__()
        self.csp1 = CSPBlock(3, 64, 1)
        self.csp2 = CSPBlock(64, 128, 2)
        self.csp3 = CSPBlock(128, 256, 8)
        self.csp4 = CSPBlock(256, 512, 8)
        self.csp5 = CSPBlock(512, 1024, 4)
        self.spp = SPP(1024, 1024)  # SPP Layer 추가

    def forward(self, x):
        out1 = self.csp1(x)
        out2 = self.csp2(out1)
        out3 = self.csp3(out2)
        out4 = self.csp4(out3)
        out5 = self.csp5(out4)
        spp_out = self.spp(out5)  # SPP 결과 반환
        return out3, out4, spp_out  # 다양한 크기의 피처 맵 반환



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

import torch
import numpy as np

class YOLOPostProcessor:
    def __init__(self, conf_threshold=0.5, iou_threshold=0.4, input_dim=416):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_dim = input_dim

    def process_predictions(self, predictions, original_img_shape):
        boxes = self.filter_boxes(predictions)
        boxes = self.rescale_boxes(boxes, original_img_shape)
        boxes = self.non_max_suppression(boxes)
        return boxes

    def filter_boxes(self, predictions):
        mask = predictions[..., 4] > self.conf_threshold
        predictions = predictions[mask]
        boxes = []
        for pred in predictions:
            x, y, w, h = pred[:4] * self.input_dim
            conf = pred[4]
            class_scores = pred[5:]
            class_id = class_scores.argmax().item()
            class_score = class_scores[class_id].item()
            boxes.append([x, y, w, h, conf, class_id, class_score])
        return boxes

    def rescale_boxes(self, boxes, original_img_shape):
        orig_h, orig_w = original_img_shape
        scale_factor = min(self.input_dim / orig_w, self.input_dim / orig_h)
        for box in boxes:
            box[0] = (box[0] - (self.input_dim - scale_factor * orig_w) / 2) / scale_factor
            box[1] = (box[1] - (self.input_dim - scale_factor * orig_h) / 2) / scale_factor
            box[2] /= scale_factor
            box[3] /= scale_factor
        return boxes

    def non_max_suppression(self, boxes):
        if len(boxes) == 0:
            return []
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        nms_boxes = []
        while boxes:
            chosen_box = boxes.pop(0)
            nms_boxes.append(chosen_box)
            boxes = [box for box in boxes if self.calculate_iou(chosen_box, box) < self.iou_threshold]
        return nms_boxes

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)    

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

