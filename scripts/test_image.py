from densEstAI import inference_image

if __name__=="__main__":
    inference_image(
        # "results/train/weights/best.pt",
        "SCUT_HEAD.pt",
        "datasets/test/SCUT-HEAD", 
        "results/predict/SCUT-HEAD-self"
        )