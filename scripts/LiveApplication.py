import queue
import threading
import time
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from scripts.modules.Density import DensityManager
from scripts.modules.Pyplot import PlotManager
from scripts.modules.DataBase import DatabaseManager
from scripts.modules.Email import EmailManager
from scripts.modules.FTP import FTPmanager
from scripts.modules.HTML import HtmlManager
from scripts.yolov5 import YOLOTrainer

class ApplicationHandler:
    def __init__(self, weight_path, db_config, email_config, ftp_config):
        # 카메라 및 모델 초기화
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 30
        self.rawCapture = PiRGBArray(self.camera, size=(640, 480))

        # Model, Density 및 Pyplot 초기화
        self.model = YOLOTrainer(weight_path=weight_path)
        self.density_manager = DensityManager(self.frame_height)
        self.plot_manager = PlotManager()
        
        # 큐로 프레임 데이터 관리 (최대 5개)
        self.frame_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()

        self.database_manager = DatabaseManager(**db_config)
        self.email_manager = EmailManager(**email_config)
        self.ftp_manager = FTPmanager(**ftp_config)
        self.html_manager = HtmlManager()
        
        
    def camera_capture(self):
        """카메라에서 이미지를 캡처하여 큐에 추가"""
        self.database_manager.insert_progresslogs(
            process_name="camera_capture", step_name="capture per frames", 
            status="started", details="Initializing camera capture loop")

        for frame in self.camera.capture_continuous(
                self.rawCapture, format="bgr", use_video_port=True):
            if self.stop_event.is_set():
                break
            image = frame.array

            # 큐가 가득 찼을 경우 대기
            try:
                self.frame_queue.put(image, timeout=1)  # 큐에 이미지 추가
            except queue.Full:
                self.database_manager.insert_errorlogs(
                    error_type="queue.Full", message="Queue is full, skipping frame")

            # 화면에 표시
            cv2.imshow("Live Capture", image)
            self.rawCapture.truncate(0)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                self.stop_event.set()
                break

        self.database_manager.insert_progresslogs(
            process_name="camera_capture", step_name="capture per frames",
            status="completed", details="Camera capture loop terminated")


    def process_frames(self):
        """큐에서 프레임을 가져와 처리"""
        self.log_progress(
            process_name="process_frames", step_name="frame processing",
            status="started", details="Initializing frame processing loop")

        while not self.stop_event.is_set():
            try:
                # 큐에서 프레임 가져오기
                image = self.frame_queue.get(timeout=1)

                # YOLO 모델로 예측 (예: 밀도 계산)
                predictions, _ = self.model.predict(image=image, return_raw=True)
                density = self.density_manager.calculate_density(predictions)

                # 밀도 값을 실시간 그래프에 업데이트
                self.plot_manager.update_Live_pyplot(density)

                self.log_progress(
                    process_name="process_frames", step_name="frame processed",
                    status="in progress", details=f"Density calculated: {density:.2f}")
                        
            except queue.Empty:
                self.database_manager.insert_errorlogs(
                    error_type="queue.Empty", message="Queue is empty, waiting for frames...")
                time.sleep(0.1)
            except Exception as e:
                self.database_manager.insert_errorlogs(
                    error_type="processing_error", message=f"Unexpected error during frame processing: {e}")
                break

        self.log_progress(
            process_name="process_frames", step_name="frame processing",
            status="completed", details="Frame processing loop terminated")

    def log_progress(self, step_name, status, details, density=None):
        """로깅 작업 통합"""
        self.database_manager.insert_progresslogs(
            process_name="process_frames", step_name=step_name, status=status, details=details)
        self.html_manager.append_html(
            process_name="process_frames", step_name=step_name, status=status, details=details, density=density)

    def app_start_running(self):
        """애플리케이션 실행"""
        self.database_manager.insert_progresslogs(
            process_name="Run", step_name="threads started",
            status="started", details="Starting capture and processing threads")

        capture_thread = threading.Thread(target=self.camera_capture, daemon=True)
        process_thread = threading.Thread(target=self.process_frames, daemon=True)

        try:
            # 스레드 시작
            capture_thread.start()
            process_thread.start()

            # 스레드 종료 대기
            capture_thread.join()
            process_thread.join()

            self.database_manager.insert_progresslogs(
                process_name="Run", step_name="threads completed",
                status="completed", details="All threads completed successfully")
        except Exception as e:
            error_message = f"Error in thread execution: {e}"
            self.database_manager.insert_errorlogs(
                error_type="thread_error", message=error_message)
            self.email_manager.SendEmail(
                    subject="Error Notification: Application Processing", body=error_message)
        finally:
            self.stop_event.set()  # 모든 스레드 종료 신호
            self.camera.close()  # PiCamera 리소스 해제
            cv2.destroyAllWindows() # OpenCV 윈도우 종료
            self.database_manager.insert_progresslogs(
                process_name="Run", step_name="application terminated",
                status="completed", details="Application has stopped")
            print("Application stopped.")

# 실행 예제
if __name__ == "__main__":
    db_config = {"host": "localhost", "user": "your_user", 
                 "password": "your_password", "database": "your_database"}
    email_config = {"sender_email": "your_email@gmail.com",
                    "sender_password": "your_email_password", 
                    "recipient_email": "recipient_email@gmail.com"}
    ftp_config = {"ftp_server":"ftp_server", 
                  "ftp_user":"user_nmae", "ftp_password":"password"}

    DensityAI_LiveApplication = ApplicationHandler(weight_path="path/to/weights", 
                                                   db_config=db_config, email_config=email_config, ftp_config=ftp_config)
    DensityAI_LiveApplication.app_start_running()
