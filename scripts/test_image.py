import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import run_inference 

if __name__=="__main__":
    run_inference(
        # "results/train/weights/best.pt",
        "SCUT_HEAD.pt",
        "datasets/test/SCUT-HEAD", 
        "results/predict/SCUT-HEAD-self"
        )