import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from densEstAI.core import start_stream  
 
if __name__=='__main__':
    start_stream(
        video_path="./datasets/test/video/test11.mp4", 
        model_path="./results/train34/weights/best.pt",
        # model_path="scut_head.pt",
        output_name="predict_Msize.mp4",
        camera_height=2.7
        )
    
 