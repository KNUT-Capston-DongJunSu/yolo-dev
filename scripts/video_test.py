import os
import sys
import time 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from densEstAI.core import ThreadedVideoStreamer
 
if __name__=='__main__':
    streamer = ThreadedVideoStreamer(
        video_path="./datasets/test/video/test11.mp4", 
        #model_path="results/medium.pt",
        model_path="scut_head.pt",
        output_name="predict_scut.mp4",
        camera_height=2.7
        )
    
    try: 
        streamer.start_stream()
        
    except Exception as e:
        print(f"[ERROR]: {e}")
    
    finally:
        streamer.stop_stream()

 