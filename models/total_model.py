import torch
import torch.nn as nn
from torchvision.ops import nms

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

class YOLOPostProcessor:
    def __init__(self, conf_threshold=0.5, iou_threshold=0.4, input_dim=416):
        """
        YOLO 모델의 추론 결과를 처리하는 PostProcessor.
        
        Args:
            conf_threshold (float): 신뢰도 점수 임계값 (default=0.5)
            iou_threshold (float): NMS에 사용할 IoU 임계값 (default=0.4)
            input_dim (int): 모델 입력 크기 (default=416)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_dim = input_dim

    def process_predictions(self, predictions, original_img_shapes):
        """
        YOLO 모델의 원본 예측 결과를 후처리하여 최종 바운딩 박스 생성.
        
        Args:
            predictions (torch.Tensor): YOLO 모델의 출력. Shape: [batch_size, num_predictions, num_classes + 5]
            original_img_shape (tuple): 원본 이미지 크기 (height, width)
        
        Returns:
            list: 최종 바운딩 박스 리스트. 각 박스는 [x, y, w, h, confidence, class_id].
        """
        batch_boxes = []
        for batch_idx, pred in enumerate(predictions):
            filtered_boxes = self.filter_boxes(pred)
            rescaled_boxes = self.rescale_boxes(filtered_boxes, original_img_shapes[batch_idx])
            nms_boxes = self.non_max_suppression(rescaled_boxes)
            batch_boxes.append(nms_boxes)
        return batch_boxes


    def filter_boxes(self, predictions):
        """
        신뢰도 점수가 일정 기준 이상인 바운딩 박스를 필터링.
        
        Args:
            predictions (torch.Tensor): 예측 결과.
        
        Returns:
            list: 필터링된 바운딩 박스.
        """
        mask = predictions[..., 4] > self.conf_threshold  # confidence score > threshold
        filtered_preds = predictions[mask]
        boxes = []
        for pred in filtered_preds:
            x, y, w, h = pred[:4]  # 바운딩 박스 좌표
            conf = pred[4]  # 신뢰도
            class_scores = pred[5:]  # 클래스 점수
            class_id = class_scores.argmax().item()  # 가장 높은 점수의 클래스
            class_score = class_scores[class_id].item()  # 해당 클래스의 점수
            boxes.append([x, y, w, h, conf, class_id, class_score])
        return boxes

    def rescale_boxes(self, boxes, original_img_shape):
        """
        모델 입력 크기에서 원본 이미지 크기로 바운딩 박스 크기를 조정.
        
        Args:
            boxes (list): 바운딩 박스 리스트.
            original_img_shape (tuple): 원본 이미지 크기 (height, width).
        
        Returns:
            list: 원본 이미지 크기에 맞게 조정된 바운딩 박스 리스트.
        """
        orig_h, orig_w = original_img_shape
        scale_factor = min(self.input_dim / orig_w, self.input_dim / orig_h)
        for box in boxes:
            box[0] = (box[0] - (self.input_dim - scale_factor * orig_w) / 2) / scale_factor  # x
            box[1] = (box[1] - (self.input_dim - scale_factor * orig_h) / 2) / scale_factor  # y
            box[2] /= scale_factor  # w
            box[3] /= scale_factor  # h
        return boxes

    def non_max_suppression(self, boxes):
        """
        Non-Maximum Suppression (NMS) 알고리즘을 통해 중복 바운딩 박스 제거.
        
        Args:
            boxes (list): 바운딩 박스 리스트.
        
        Returns:
            list: NMS를 통과한 바운딩 박스 리스트.
        """
        if len(boxes) == 0:
            return []

        boxes_tensor = torch.tensor(boxes)
        xywh_boxes = boxes_tensor[:, :4]  # x, y, w, h
        scores = boxes_tensor[:, 4]  # confidence scores
        indices = nms(xywh_boxes, scores, self.iou_threshold)
        return boxes_tensor[indices].tolist()

    def calculate_iou(self, box1, box2):
        """
        두 바운딩 박스 간 IoU(Intersection over Union) 계산.
        
        Args:
            box1 (list): 첫 번째 박스 [x, y, w, h, ...].
            box2 (list): 두 번째 박스 [x, y, w, h, ...].
        
        Returns:
            float: IoU 값.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)  # 교집합 영역
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]

        return inter_area / (box1_area + box2_area - inter_area + 1e-6)  # IoU 계산
