import cv2
import os
import torch
from ocsort import OCSort
from queue import Empty, Full
from multiprocessing import Process, Queue
from src.app.Density import DensityManager
from src.app.Pyplot import PlotManager
from src.app.yolo_utils import predict_yolo
from src.app.utils import custom_plot
from src.app.utils import update_tracks

class VideoStreamHandler:
    def __init__(self, video_path, model_path, output_video):
        self.video_path = video_path
        self.model_path = model_path
        self.output_video = output_video
        save_dir = os.path.dirname(self.output_video)
        os.makedirs(save_dir, exist_ok=True)

        self.frame_queue = None
        self.result_queue = None        
        self.frame_count = 0  # 프레임 카운터

        self.tracker = OCSort(  # OCSort 객체 초기화
            det_thresh=0.3,  
            max_age=30,
            min_hits=3
        )

    def process_frames(self):
        """프레임을 YOLO로 처리하고 결과를 큐에 저장"""
        while True:
            frame, frame_id = self.frame_queue.get()
            if frame is None:  
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = predict_yolo(
                model_path=self.model_path,
                frame=rgb_frame,
                save=False,
                half=True,
                stream=False
            )

            # YOLO 예측 결과 -> SORT 트래커에 입력
            boxes = results['prediction']['boxes']
            confidences = results['prediction']['scores']  # 실제 confidence 사용
            class_ids = results['prediction']['classes']  # 실제 클래스 아이디 사용
            
            data_list = [box + [conf, cls] for box, conf, cls in zip(boxes, confidences, class_ids)]
            tracker_input = torch.tensor(data_list, dtype=torch.float32)

            # SORT 업데이트 및 트래킹 결과 받기
            if tracker_input.shape[0] == 0:
                tracked_objects = []  # 또는 빈 텐서 등, 트래커가 처리할 수 없는 빈 입력에 대비
            else:
                tracked_objects = self.tracker.update(tracker_input, frame_id)
                # filtered_ids = self.update_tracks(tracked_objects)
                # if len(filtered_ids) != 0:
                #     tracked_ids = tracked_objects[:, 4].astype(int)
                #     indices = np.where(np.isin(tracked_ids, filtered_ids))[0]
                #     tracked_objects = tracked_objects[indices] 
                 
            # density = self.density_manager.calculate_density(results["prediction"])
            density = None
            plot = custom_plot(frame, tracked_objects)  # Bounding box 그리기
            self.result_queue.put({'plot': plot, 'density': density})

    def start_stream(self, camera_height):
        # 각 프로세스가 독립적이기 때문에 
        # 데이터를 주고받으려면 공유 가능한 큐를 미리 생성해야 함
        self.frame_queue = Queue(maxsize=20)
        self.result_queue = Queue(maxsize=20)
        
        process = Process(target=self.process_frames)
        process.start()
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video file: {self.video_path}")
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.graph_update_interval = max(1, fps // 2)  # 0.5초 간격으로 그래프 업데이트
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = cv2.VideoWriter(self.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        self.density_manager = DensityManager(frame_height, camera_height)
        self.pyplot_manager = PlotManager(fps)

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                if self.frame_count % 10 != 0:
                    continue
                
                try:
                    self.frame_queue.put((frame, self.frame_count), timeout=1)
                except Full:
                    print("[WARNING] Frame queue full. Dropping frame.")

                # 처리된 결과를 받아 저장
                while not self.result_queue.empty():
                    try:
                        result = self.result_queue.get(timeout=1)
                    except Empty:
                        continue
                    if result is None:
                        break
                    self.video_writer.write(result['plot'])
                    cv2.imshow("YOLO Stream", result['plot'])

                    # if self.frame_count % self.graph_update_interval == 0:
                        # self.pyplot_manager.update_Live_pyplot(result['density'])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"[ERROR] {e}")

        finally:
            # 종료 신호는 여기가 적절
            self.frame_queue.put(None)
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
            self.close_stream()

    def close_stream(self):
        print("Releasing resources...")
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
        except Exception as e:
            print(f"Queue cleanup error: {e}")
        self.frame_queue.close()
        self.result_queue.close()
        self.frame_queue.join_thread()
        self.result_queue.join_thread()
        self.pyplot_manager.close()
        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        print("Processing complete. Exiting program...")

