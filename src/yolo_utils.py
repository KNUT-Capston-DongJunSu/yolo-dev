import torch
from ultralytics import YOLO

def train_yolo(
    model_path, 
    config_path, 
    epochs=100,
    imgsz=640,
    batch=16,
    project='results',
    name=None,
    lr0=0.01,
    optimizer='SGD',
    **kwargs
    ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path).to(device)
    return model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        lr0=lr0,
        optimizer=optimizer,
        device=device,
        **kwargs
    )

def predict_yolo(
    model_path, 
    frame,
    stream=False,
    imgsz=640,
    conf=0.5,
    iou=0.7,
    device=None,
    max_det=300,
    **kwargs
    ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path).to(device)
    result = model.predict(
        source=frame,
        stream=stream,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        max_det=max_det,
        **kwargs
        )[0]
    target_cls = 0

    boxes = result.boxes
    filtered_boxes = boxes[boxes.cls == target_cls]
    
    boxes = filtered_boxes.xyxy
    scores = filtered_boxes.conf
    classes = filtered_boxes.cls

    if isinstance(boxes, torch.Tensor): boxes = boxes.tolist()
    if isinstance(scores, torch.Tensor): scores = scores.tolist()
    if isinstance(classes, torch.Tensor): classes = classes.tolist()
 
    plot = result.plot()
    
    return {"prediction": {
                "boxes": boxes,
                "scores": scores,
                "classes": classes
            },
            "plot": plot}

   
    