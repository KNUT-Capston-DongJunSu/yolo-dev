import cv2
import os
import torch
from ocsort import OCSort
from densEstAI.core.plotter import DensityPlotter
from densEstAI.yolo.yolo_manager import YoloManager
from densEstAI.core.density_estimator import DensityEstimator
from densEstAI.utils.common import get_best_model
from densEstAI.utils.tracking import filter_tracks_by_class 
from densEstAI.utils.drawing_boxes import draw_tracking_boxes

class VideoStreamer:
    def __init__(self, video_path, model_path, camera_height=3.0):
        self.video_path = video_path
        self.model_path = model_path

        self.frame_queue = None
        self.result_queue = None    
        self.model = None
        self.density_manager = None 
        self.pyplot_manager = None    
        self.frame_id = 0  # 프레임 카운터
        self.camera_height = camera_height

        self.tracker = OCSort(det_thresh=0.3, max_age=30, min_hits=3)
        self.model = YoloManager(self.model_path)
    
    def start_stream(self, output_path="results/predict/video/predict.mp4"):
        save_dir = os.path.dirname(output_path)
        os.makedirs(save_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {self.video_path}")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.graph_update_interval = max(1, fps // 2)  # 0.5초 간격으로 그래프 업데이트
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        if self.density_manager is None or self.pyplot_manager is None:
            self.density_manager = DensityEstimator(frame_height, self.camera_height)
            self.pyplot_manager = DensityPlotter(fps) 

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_id += 1
                # if self.frame_id % 3 != 0:
                #     continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model.smart_predict_yolo(
                    frame=rgb_frame,
                    conf=0.07,
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
                    tracked_objects = self.tracker.update(tracker_input, self.frame_id)
                    # filtered_ids = self.filter_tracks_by_class(tracked_objects)
                    # if len(filtered_ids) != 0:
                    #     tracked_ids = tracked_objects[:, 4].astype(int)
                    #     indices = np.where(np.isin(tracked_ids, filtered_ids))[0]
                    #     tracked_objects = tracked_objects[indices] 
                # density = self.density_manager.calculate_density(results["prediction"])
                density = None
                plot = draw_tracking_boxes(frame, tracked_objects)  # Bounding box 그리기

                video_writer.write(plot)
                cv2.imshow("YOLO Stream", plot)

                # if self.frame_id % self.graph_update_interval == 0:
                    # self.pyplot_manager.update_Live_pyplot(result['density'])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"[ERROR] {e}")

        finally:
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()