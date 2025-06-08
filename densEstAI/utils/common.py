import os
from PIL import Image

def img_shape(image_path):
    img = Image.open(image_path)
    width, height = img.size
    print(f"이미지 크기: {width}x{height}")

def get_best_model(weights_dir):
    best_model = os.path.join(weights_dir, "best.pt")
    return best_model if os.path.exists(best_model) else None

def detect_display():
    return "DISPLAY" in os.environ and os.environ["DISPLAY"]