import torch 
from torchvision.ops import nms

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