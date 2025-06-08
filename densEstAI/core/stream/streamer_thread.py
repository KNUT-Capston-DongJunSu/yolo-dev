import cv2
import os
import torch
import threading
import queue
from ocsort import OCSort
import numpy as np
from densEstAI.core.plot.plotter import update_live_density
from densEstAI.core.yolo.yolo_manager import YoloManager
from densEstAI.core.stream.density_estimation import calculate_density
from densEstAI.core.utils.tracking import tracking_object
from densEstAI.core.utils.drawing_boxes import draw_tracking_boxes
from densEstAI.utils.common import detect_display

def reader_thread(cap, frame_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    frame_queue.put(None)

def processor_thread(frame_queue, output_queue, model, tracker, density_manager, pyplot_manager):
    frame_id = 0
    while True:
        frame = frame_queue.get()
        if frame is None:
            output_queue.put(None)
            break

        frame_id += 1
        results = model.smart_predict_yolo(frame, conf=0.07, save=False, half=True, stream=False)
        tracked_objects = tracking_object(tracker, results, frame_id)
        density = density_manager.calculate_density(results)
        visual_frame = draw_tracking_boxes(frame, tracked_objects)

        pyplot_manager.update_Live_pyplot(density)
        output_queue.put(visual_frame)

def writer_thread(output_queue, video_writer, is_display):
    while True:
        plot = output_queue.get()
        if plot is None:
            break
        video_writer.write(plot)

        if is_display:
            cv2.imshow("YOLO Stream", plot)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main(video_path, model_path, output_path="results/predict/video/predict.mp4", camera_height=3.0):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # 모델/관리자 초기화
    model = YoloManager(model_path)
    tracker = OCSort(det_thresh=0.3, max_age=30, min_hits=3)
    density_manager = DensityEstimator(frame_height, camera_height)
    pyplot_manager = DensityPlotter(fps)

    # 큐 선언
    frame_queue = queue.Queue(maxsize=5)
    output_queue = queue.Queue(maxsize=5)

    # 쓰레드 정의 및 시작
    threads = [
        threading.Thread(target=reader_thread, args=(cap, frame_queue)),
        threading.Thread(target=processor_thread, args=(frame_queue, output_queue, model, tracker, density_manager, pyplot_manager)),
        threading.Thread(target=writer_thread, args=(output_queue, video_writer, detect_display()))
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()