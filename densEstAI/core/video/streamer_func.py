import cv2
import os
from ocsort import OCSort
from densEstAI.core.plotter import DensityPlotter
from densEstAI.yolo.yolo_manager import YoloManager
from densEstAI.core.density_estimator import DensityEstimator
from densEstAI.utils.drawing_boxes import draw_tracking_boxes

def start_stream(video_path, model_path, output_path="results/predict/video/predict.mp4", camera_height=3.0, scale=1):
    track_hist = []
    frame_id = 0
    save_dir = os.path.dirname(output_path)
    os.makedirs(save_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # graph_update_interval = max(1, fps // 2) 
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    tracker = OCSort(det_thresh=0.3, max_age=30, min_hits=3)
    model = YoloManager(model_path)
    density_manager = DensityEstimator(camera_height, frame_height)
    pyplot_manager = DensityPlotter(fps) 

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            # if frame_id % 3 != 0:
            #     continue

            results = model.smart_predict_yolo(frame=frame, conf=0.07, save=False, half=True, stream=False)

            if results.shape[0] == 0:
                track_hist = []
            else:
                if len(results) < len(track_hist):
                    track_hist = tracker.update(track_hist, frame_id)
                else:
                    track_hist = tracker.update(results, frame_id)
                    # tracked_objects = tracking_object(tracker, results, frame_id)
            
            density = density_manager.calculate_density(results)
            plot = draw_tracking_boxes(frame, track_hist)  # Bounding box 그리기

            video_writer.write(plot)
            scaled_frame = cv2.resize(plot, (int(1920/2), int(1080/2)))
            cv2.imshow("YOLO Stream", scaled_frame)

            # if frame_id % graph_update_interval == 0:
            pyplot_manager.update_Live_pyplot(density)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

