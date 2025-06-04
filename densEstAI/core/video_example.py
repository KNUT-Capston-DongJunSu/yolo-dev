import cv2
import os
import torch
from ocsort import OCSort
from queue import Empty, Full
from multiprocessing import Process, Queue
from densEstAI.core.calc_density import DensityManager
from densEstAI.core.plot_dens import PlotManager
from densEstAI.core.yolo_api import YoloAPI
from densEstAI.core.utils import draw_tracking_boxes
from densEstAI.core.utils import filter_tracks_by_class 
from densEstAI.core.utils import get_best_model

class VideoStreamHandler:
    def __init__(self, video_path, model_path="results/train/weights", camera_height=3.0):
        self.video_path = video_path

        self.frame_queue = None
        self.result_queue = None    
        self.density_manager = None 
        self.pyplot_manager = None    
        self.frame_count = 0  # 프레임 카운터
        self.camera_height = camera_height

        self.tracker = OCSort(det_thresh=0.3, max_age=30, min_hits=3)
        self.model = YoloAPI(get_best_model(model_path))

    def process_frames(self):
        """프레임을 YOLO로 처리하고 결과를 큐에 저장"""
        while True:
            frame, frame_id = self.frame_queue.get()
            if frame is None:  
                self.result_queue.put(None)
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model.smart_predict_yolo(
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
                # filtered_ids = self.filter_tracks_by_class(tracked_objects)
                # if len(filtered_ids) != 0:
                #     tracked_ids = tracked_objects[:, 4].astype(int)
                #     indices = np.where(np.isin(tracked_ids, filtered_ids))[0]
                #     tracked_objects = tracked_objects[indices] 
                 
            # density = self.density_manager.calculate_density(results["prediction"])
            density = None
            plot = draw_tracking_boxes(frame, tracked_objects)  # Bounding box 그리기
            self.result_queue.put({'plot': plot, 'density': density})

    def start_stream(self, output_path="results/predict/video/predict.mp4"):
        save_dir = os.path.dirname(output_path)
        os.makedirs(save_dir, exist_ok=True)

        # 각 프로세스가 독립적이기 때문에 
        # 데이터를 주고받으려면 공유 가능한 큐를 미리 생성해야 함
        self.frame_queue = Queue(maxsize=20)
        self.result_queue = Queue(maxsize=20)
        
        process = Process(target=self.process_frames)
        process.start()
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {self.video_path}")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.graph_update_interval = max(1, fps // 2)  # 0.5초 간격으로 그래프 업데이트
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        if self.density_manager is None or self.pyplot_manager is None:
            self.density_manager = DensityManager(frame_height, self.camera_height)
            self.pyplot_manager = PlotManager(fps)  

        try:
            while cap.isOpened():
                ret, frame = cap.read()
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
                    video_writer.write(result['plot'])
                    cv2.imshow("YOLO Stream", result['plot'])

                    # if self.frame_count % self.graph_update_interval == 0:
                        # self.pyplot_manager.update_Live_pyplot(result['density'])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"[ERROR] {e}")

        finally:
            self.frame_queue.put(None)
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
            self.frame_queue.close()
            self.result_queue.close()
            self.frame_queue.join_thread()
            self.result_queue.join_thread()
            self.pyplot_manager.close()
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()
            print("Processing complete. Exiting program...")
        
