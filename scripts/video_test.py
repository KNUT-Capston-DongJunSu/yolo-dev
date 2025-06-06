import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from densEstAI.core import VideoStreamer
from densEstAI.core import start_stream

def main_process():
    video_handler = VideoStreamer(video_path="./datasets/test/video/test3.mp4", model_path="./results/train/weights/best.pt", camera_height=3.0)
    video_handler.start_stream(output_path="./results/predict/video/our_model_default.mp4")

if __name__=='__main__':
    # main_process()
    start_stream(video_path="./datasets/test/video/test3.mp4", model_path="./results/train15/weights/best.pt", output_path="./results/predict/video/our_model_default.mp4", camera_height=2.7,scale=2)