from densEstAI import predict_img

if __name__=="__main__":
    predict_img(
        # "results/train/weights/best.pt",
        "SCUT_HEAD.pt",
        "datasets/test/SCUT-HEAD", 
        "results/predict/SCUT-HEAD-self"
        )