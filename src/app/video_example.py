import cv2
import os
import signal
import sys
import torch
import numpy as np
from ocsort import OCSort
from ultralytics import YOLO
from PIL import Image, ImageDraw
from multiprocessing import Process, Queue
from src.app.Density import DensityManager
from src.app.Pyplot import PlotManager
from src.yolo_utils import predict_yolo

class VideoStreamHandler:
    def __init__(self, video_path, model_path, save_dir, output_video):
        self.video_path = video_path
        self.model_path = model_path
        self.scale = 0.5  # 비율 설정

        # YOLO 모델, 밀도 관리자
        self.density_manager = DensityManager(frame_height=480, camera_height=3.0)
        self.pyplot_manager = PlotManager()
        self.save_dir = save_dir

        # 큐 초기화
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
    
        self.tracker = OCSort(  # OCSort 객체 초기화
            det_thresh=0.07,  # detection threshold
            iou_threshold=0.3,  # 박스 간 IoU threshold
            max_age=30,
            min_hits=3
        )

        self.output_video = output_video

    def process_frames(self):
        """프레임을 YOLO로 처리하고 결과를 큐에 저장"""
        while True:
            frame, frame_id = self.frame_queue.get()
            if frame is None:  # 종료 신호
                self.result_queue.put(None)
                break
           
            results = predict_yolo(
                model_path=self.model_path,
                frame=frame,
                imgsz=1280,
                conf=0.05,     
                save=False,
                half=True,
                stream=False
            )

            # YOLO 예측 결과 -> SORT 트래커에 입력
            boxes = results['prediction']['boxes']
            print(boxes)
            confidences = results['prediction']['scores']  # 실제 confidence 사용
            class_ids = results['prediction']['classes']  # 실제 클래스 아이디 사용
            
            data_list = [box + [conf, cls] for box, conf, cls in zip(boxes, confidences, class_ids)]
            tracker_input = torch.tensor(data_list, dtype=torch.float32)

            # SORT 업데이트 및 트래킹 결과 받기
            if tracker_input.shape[0] == 0:
                tracked_objects = []  # 또는 빈 텐서 등, 트래커가 처리할 수 없는 빈 입력에 대비
            else:
                tracked_objects = self.tracker.update(tracker_input, frame_id)

            density = self.density_manager.calculate_density(results["prediction"])
            plot_bgr = self.custom_plot(frame, tracked_objects)  # Bounding box 그리기
            self.result_queue.put((plot_bgr, density))
    
    def custom_plot(self, frame, tracked_objects):
        """Bounding box와 트래킹 ID 표시"""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, *rest  = map(int, obj)
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            draw.text((x1, y1 - 10), f"ID: {track_id}", fill="red")

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

        self.pyplot_manager.fps = fps
        
        # 프로세스 생성
        process_process = Process(target=self.process_frames)
        process_process.start()

        try:
            frame_count = 0  # 프레임 카운터
            graph_update_interval = max(1, fps // 2)  # 0.5초 간격으로 그래프 업데이트

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    self.frame_queue.put(None)  # 종료 신호
                    break

                frame_resized = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
                rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                self.frame_queue.put((rgb_frame, frame_count))
                """
                [249 160 127]
                [245 155 113]
                [248 152 110]
                [244 148 106]
                [244 144  92]
                [247 140  79]
                [247 128  64]
                [246 124  59]
                [235 127  44]
                [244 213 122]
                [250 255 201]
                [249 255 214]
                """

                # 처리된 결과를 받아 저장
                while not self.result_queue.empty():
                    result = self.result_queue.get()
                    if result is None:
                        break
                    plot_bgr, density = result
                    video_writer.write(plot_bgr)
                    cv2.imshow("YOLO Stream", plot_bgr)

                    if frame_count % graph_update_interval == 0:
                        self.pyplot_manager.update_Live_pyplot(
                            current_value=density,
                            filename=f"results/density/graph{frame_count}.png",
                        )

                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            print("Releasing resources...")

            # 비디오 및 OpenCV 종료
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()

            # 자식 프로세스 강제 종료
            if process_process.is_alive():
                print("Terminating child process...")
                process_process.terminate()
                process_process.join(timeout=5)
                if process_process.is_alive():
                    os.kill(process_process.pid, signal.SIGKILL)  # 강제 종료

            # 큐 정리
            try:
                while not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
                while not self.result_queue.empty():
                    self.result_queue.get_nowait()
            except Exception as e:
                print(f"Queue cleanup error: {e}")

            self.frame_queue.close()
            self.result_queue.close()
            self.frame_queue.cancel_join_thread()
            self.result_queue.cancel_join_thread()

            # Pyplot 리소스 정리
            self.pyplot_manager.close()
            print("Processing complete. Exiting program...")
            sys.exit(0)  # 강제 종료

