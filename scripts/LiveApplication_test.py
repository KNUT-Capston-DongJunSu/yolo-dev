import queue
import threading
import time
import cv2
from scripts.modules.Density import DensityManager
from scripts.modules.Pyplot import PlotManager
from scripts.modules.DataBase import DatabaseManager
from scripts.modules.Email import EmailManager
from scripts.modules.FTP import FTPmanager
from scripts.modules.HTML import HtmlManager
from scripts.yolov5 import YOLOTrainer


class ApplicationHandler:
    def __init__(self, weight_path, video_source=0):
        # 비디오 소스 초기화 (0 = 기본 카메라, 또는 비디오 파일 경로)
        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {self.video_source}")

        # Model, Density 및 Pyplot 초기화
        self.model = YOLOTrainer(weight_path=weight_path)
        self.density_manager = DensityManager(frame_height=480)  # 고정된 높이 값
        self.plot_manager = PlotManager()

        # 큐로 프레임 데이터 관리 (최대 5개)
        self.frame_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()

    def camera_capture(self):
        """카메라에서 이미지를 캡처하여 큐에 추가"""
        self.log_progress(
            step_name="capture per frames", status="started",
            details="Initializing camera capture loop")

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            try:
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                pass

            cv2.imshow("Live Capture", frame)

            # 'q'를 눌러 종료
            if cv2.waitKey(10) & 0xFF == ord("q"):
                self.stop_event.set()
                break

    def process_frames(self):
        while not self.stop_event.is_set():
            try:
                # 큐에서 프레임 가져오기
                frame = self.frame_queue.get(timeout=1)
                frame_resized = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

                # YOLO 모델로 예측
                # YOLO 모델로 예측
                results = self.model.predict(
                    frame=frame_resized,
                    conf_threshold=0.2,
                    iou_threshold=0.6,
                    save=False,
                    half=True  # FP16 모드 활성화
                )
                density = self.density_manager.calculate_density(results["prediction"])

                # 밀도 값을 실시간 그래프에 업데이트
                self.plot_manager.update_Live_pyplot(density)
            except:
                pass

    def app_start_running(self):
        """애플리케이션 실행"""

        capture_thread = threading.Thread(target=self.camera_capture, daemon=True)
        process_thread = threading.Thread(target=self.process_frames, daemon=True)

        try:
            # 스레드 시작
            capture_thread.start()
            process_thread.start()

            # 스레드 종료 대기
            capture_thread.join()
            process_thread.join()
        except Exception as e:
            pass
        finally:
            self.stop_event.set()
            self.cap.release()  # VideoCapture 리소스 해제
            cv2.destroyAllWindows()  # OpenCV 윈도우 종료
            print("Application stopped.")


# 실행 예제
if __name__ == "__main__":

    DensityAI_LiveApplication = ApplicationHandler(
        weight_path="path/to/weights",
        video_source=0  # 0: 기본 카메라, 또는 비디오 파일 경로
    )
    DensityAI_LiveApplication.app_start_running()
