import cv2
import os
import numpy as np
from multiprocessing import Process, Queue
from modules.Density import DensityManager
from modules.Pyplot import PlotManager
from yolov5 import YOLOTrainer
from PIL import Image, ImageDraw

class VideoStreamHandler:
    def __init__(self, video_path, weight_path, save_dir, output_video):
        self.video_path = video_path
        self.scale = 0.5  # 비율 설정
        self.fps = 30

        # YOLO 모델, 밀도 관리자
        self.trainer = YOLOTrainer(weight_path=weight_path)
        self.density_manager = DensityManager(frame_height=480, camera_height=3.0)
        self.pyplot_manager = PlotManager()
        self.save_dir = save_dir

        # 큐 초기화
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)

        self.output_video = output_video

    def process_frames(self):
        """프레임을 YOLO로 처리하고 결과를 큐에 저장"""
        while True:
            frame = self.frame_queue.get()
            if frame is None:  # 종료 신호
                self.result_queue.put(None)
                break

            results = self.trainer.predict(
                frame=frame,
                conf_threshold=0.10,
                iou_threshold=0.6,
                save=False,
                half=True
            )
            density = self.density_manager.calculate_density(results["prediction"])
            plot_bgr = self.custom_plot(frame, results)  # Bounding box 그리기
            self.result_queue.put((plot_bgr, density))

    def custom_plot(self, frame, results):
        """Bounding box만 표시"""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        boxes = results["prediction"]["boxes"]

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)

        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def start_stream(self):
        """메인 프로세스에서 비디오 읽기 및 결과 저장"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {self.video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale)
        video_writer = cv2.VideoWriter(self.output_video, cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps, (frame_width, frame_height))

        # 프로세스 생성
        process_process = Process(target=self.process_frames)
        process_process.start()

        # 프레임 읽기 및 처리 결과 저장
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                self.frame_queue.put(None)  # 종료 신호
                break

            frame_resized = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
            self.frame_queue.put(frame_resized)

            # 처리된 결과를 받아 저장
            while not self.result_queue.empty():
                result = self.result_queue.get()
                if result is None:
                    break
                plot_bgr, density = result
                video_writer.write(plot_bgr)  # 메인 프로세스에서 저장
                cv2.imshow("YOLO Stream", plot_bgr)

                # 그래프 데이터 추가
                self.pyplot_manager.update_Live_pyplot(
                    current_value=density,
                    filename=f"results/density/graph{density}.png"
                )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        process_process.join()

if __name__ == "__main__":
    video_path = "datasets/video/ski.mp4"
    weight_path = "results/train_preprocessing/weights/best.pt"
    save_dir = "results/predict/"
    output_video = "results/predict/ski_predict.mp4"

    os.makedirs(save_dir, exist_ok=True)
    video_handler = VideoStreamHandler(video_path, weight_path, save_dir, output_video)
    video_handler.start_stream()
