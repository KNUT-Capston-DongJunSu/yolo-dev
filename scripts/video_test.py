import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from densEstAI.core.video_example import VideoStreamHandler

def main_process():
    video_handler = VideoStreamHandler(video_path="./datasets/test/video/test3.mp4", model_path="./results/train/weights/best.pt", camera_height=3.0)
    video_handler.start_stream(output_path="./results/predict/video/our_model_default.mp4")

if __name__=='__main__':
    main_process()