import cv2
import os
from ocsort import OCSort
from densEstAI.core.plotter import DensityPlotter
from densEstAI.yolo.yolo_manager import YoloManager
from densEstAI.core.density_estimator import DensityEstimator
from densEstAI.utils.drawing_boxes import draw_tracking_boxes
from densEstAI.utils.tracking import tracking_object

scale = 1
output_path="results/predict/video/predict.mp4"
save_dir = os.path.dirname(output_path)
os.makedirs(save_dir, exist_ok=True)
resize_width = 960
resize_height = 540

def initalize_object(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    return cap, video_writer, fps, frame_width, frame_height
    
def app_core(model, tracker, density_manager, pyplot_manager, frame, track_hist, frame_id):
    results = model.smart_predict_yolo(frame=frame, conf=0.07, save=False, half=True, stream=False)
    track_hist = tracking_object(tracker, results, track_hist, frame_id)
    density = density_manager.calculate_density(results)
    plot = draw_tracking_boxes(frame, track_hist)  # Bounding box 그리기
    pyplot_manager.update_Live_pyplot(density)

    return plot, track_hist
    
def start_stream(video_path, model_path, camera_height):
    frame_id = 0
    track_hist = []

    cap, video_writer, fps, frame_width, frame_height = initalize_object(video_path)

    tracker = OCSort(det_thresh=0.3, max_age=30, min_hits=3)
    model = YoloManager(model_path)
    density_manager = DensityEstimator(frame_height, camera_height)
    pyplot_manager = DensityPlotter(fps)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            frame_id += 1
            if not ret:
                break
            plot, track_hist = app_core(model, tracker, density_manager, pyplot_manager, frame, track_hist, frame_id)
            video_writer.write(plot)
            cv2.imshow("YOLO Stream", (resize_width, resize_height))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

