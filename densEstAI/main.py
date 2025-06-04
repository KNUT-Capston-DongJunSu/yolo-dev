from .core.video_example import VideoStreamHandler

if __name__=="__main__":
    video_path = "datasets/test/video/test1132.mp4"
    model_path = "results/train/weights/best.pt"
    # model_path = "SCUT_HEAD.pt"
    output_video = "results/predict/video/our_model_demo3.mp4"

    video_handler = VideoStreamHandler(video_path, model_path, output_video)
    video_handler.start_stream(3)