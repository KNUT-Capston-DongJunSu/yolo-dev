<<<<<<< HEAD
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

=======
from densEstAI.utils import move_to_root_path
move_to_root_path()
>>>>>>> 0f57c93f52dcd4d3a1eb8133836acdf90a09c67a
from densEstAI.utils import run_inference 

if __name__=="__main__":
    run_inference(
        # "results/train/weights/best.pt",
        "SCUT_HEAD.pt",
        "datasets/test/SCUT-HEAD", 
        "results/predict/SCUT-HEAD-self"
        )