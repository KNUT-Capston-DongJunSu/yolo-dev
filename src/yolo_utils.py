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
    conf=0.25,
    iou=0.45,
    device=None,
    max_det=300,
    **kwargs
    ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path).to(device)
    result =  model.predict(
        source=frame,
        stream=stream,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        max_det=max_det,
        **kwargs
        )[0]

    boxes = result.boxes.xyxy
    scores = result.boxes.conf
    classes = result.boxes.cls

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
    
if __name__=="__main__":
    train_yolo(model_path='models/yolov5s.pt', 
        config_path='configs/custom.yaml', 
        epochs=100, imgsz=256, batch_size=8, 
        project='results', name='train1')

   
    