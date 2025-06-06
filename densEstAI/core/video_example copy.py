import cv2
import os
import torch
from ocsort import OCSort
import numpy as np
from densEstAI.core.calc_density import DensityManager
from densEstAI.core.plot_dens import PlotManager
from densEstAI.core.yolo_api import YoloAPI
from densEstAI.core.utils import draw_tracking_boxes
from densEstAI.core.utils import tracking_object
from densEstAI.core.utils import transform_yolo2track
from densEstAI.core.utils import track_hist

def start_stream(video_path, model_path, output_path="results/predict/video/predict.mp4", camera_height=3.0):
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
    model = YoloAPI(model_path)
    density_manager = DensityManager(frame_height, camera_height)
    pyplot_manager = PlotManager(fps) 

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            # if frame_id % 3 != 0:
            #     continue

            results = model.smart_predict_yolo(frame=frame, conf=0.07, save=False, half=True, stream=False)
            transformed_results = transform_yolo2track(results, track_hist)
            if len(transformed_results) < len(track_hist):
                input = track_hist
            else:
                input = transformed_results
            tracked_objects = tracking_object(tracker, input, frame_id)
            density = density_manager.calculate_density(results)
            plot = draw_tracking_boxes(frame, tracked_objects)  # Bounding box 그리기

            video_writer.write(plot)
            cv2.imshow("YOLO Stream", plot)

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

