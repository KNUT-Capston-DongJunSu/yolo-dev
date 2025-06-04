from .core.video_example import VideoStreamHandler

def main():
    video_handler = VideoStreamHandler(video_path="datasets/test/video/test1132.mp4")
    video_handler.start_stream(camera_height=3)

if __name__=='__main__':
    main()