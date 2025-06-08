import cv2
import os
from ocsort import OCSort
import densEstAI.core.plot.plotter as plotter
import densEstAI.core.stream.density_estimation as DensEst
from densEstAI.core.yolo.yolo_manager import YoloManager
from densEstAI.core.plot.plotter import update_live_density
from densEstAI.core.stream.density_estimation import calculate_density
from densEstAI.core.utils.tracking import tracking_object
from densEstAI.core.utils.drawing_boxes import draw_tracking_boxes
from densEstAI.utils.common import detect_display

scale = 1
output_dir = "./results/predict/"
os.makedirs(output_dir, exist_ok=True)
resize_width = 960
resize_height = 540

def initalize_object(video_path, output_name):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_dir+output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    return cap, video_writer, fps, frame_width, frame_height
    
def app_core(model, tracker, frame, track_hist, frame_id):
    results = model.smart_predict_yolo(frame=frame, conf=0.5, save=False, half=True, stream=False)
    tracked_objects = tracking_object(tracker, results, frame_id)
    density = calculate_density(results)
    plot = draw_tracking_boxes(frame, tracked_objects)  # Bounding box 그리기
    update_live_density(density)

    return plot, track_hist
    
def start_stream(video_path, model_path, output_name, camera_height):
    frame_id = 0
    track_hist = []

    cap, video_writer, fps, frame_width, frame_height = initalize_object(video_path, output_name)

    tracker = OCSort(det_thresh=0.3, max_age=30, min_hits=3)
    model = YoloManager(model_path)

    plotter.video_fps = fps
    DensEst.camera_height = camera_height
    DensEst.frame_height = frame_height

    while cap.isOpened():
        ret, frame = cap.read()
        frame_id += 1
        if not ret:
            break
        plot, track_hist = app_core(model, tracker, frame, track_hist, frame_id)
        video_writer.write(plot)
        resize_plot = cv2.resize(plot, (resize_width, resize_height))
        cv2.imshow("YOLO Stream", resize_plot)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

