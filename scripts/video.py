import cv2
import time
import os
import numpy as np
from modules.Density import DensityManager
from yolov5 import YOLOTrainer
from PIL import Image, ImageDraw

class VideoStreamHandler:
    def __init__(self, video_path, weight_path, save_dir, output_video):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video file: {self.video_path}")

        self.trainer = YOLOTrainer(weight_path=weight_path)
        self.density_manager = DensityManager(frame_height=480, camera_height=3.0)
        self.save_dir = save_dir

        # VideoWriter 초기화
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)  # 크기 축소
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)
        self.video_writer = cv2.VideoWriter(output_video, fourcc, self.fps, (frame_width, frame_height))

    def custom_plot(self, frame, results):
        """
        fbox와 확률값 없이 bounding box만 표시
        """
        # OpenCV 이미지를 PIL 이미지로 변환
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)

        # 예측 결과에서 박스 가져오기
        boxes = results["prediction"]["boxes"]

        # 박스만 그리기
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)

        # PIL 이미지를 OpenCV 포맷으로 변환
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def start_stream(self):
        """동영상 파일을 스트림으로 처리"""
        frame_count = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Video ended or failed to read frame.")
                break

            frame_count += 1
            before = time.time()

            # 프레임 크기 축소
            frame_resized = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

            # YOLO 모델로 예측
            results = self.trainer.predict(
                frame=frame_resized,
                conf_threshold=0.15,
                iou_threshold=0.6,
                save=False,
                half=True  # FP16 모드 활성화
            )

            # fbox와 확률값 없는 결과 시각화
            plot_bgr = self.custom_plot(frame_resized, results)

            # 결과 쓰기
            self.video_writer.write(plot_bgr)
            cv2.imshow("YOLO Stream", plot_bgr)

            after = time.time()
            print(f"Frame {frame_count} processed in {after - before:.2f}s")

            # 'q'를 눌러 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 자원 해제
        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "datasets/video/ski.mp4"  # 동영상 파일 경로
    weight_path = "results/train_preprocessing/weights/best.pt"
    save_dir = "results/predict/"
    output_video = "results/predict/ski_predict.mp4"
    # 디렉터리 생성
    os.makedirs(save_dir, exist_ok=True)

    # 스트림 실행
    video_handler = VideoStreamHandler(
        video_path=video_path,
        weight_path=weight_path,
        save_dir=save_dir,
        output_video=output_video
    )
    video_handler.start_stream()
