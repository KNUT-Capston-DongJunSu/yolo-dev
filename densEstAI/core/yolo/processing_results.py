import torch

def process_predicted_results(result):
    boxes = result.boxes.xyxy
    confidences = result.boxes.conf
    classes = result.boxes.cls

    if isinstance(boxes, torch.Tensor): boxes = boxes.tolist()
    if isinstance(confidences, torch.Tensor): confidences = confidences.tolist()
    if isinstance(classes, torch.Tensor): classes = classes.tolist()
    
    data_list = [box + [conf, cls] for box, conf, cls in zip(boxes, confidences, classes)]
    results = torch.tensor(data_list, dtype=torch.float32)
    return results
