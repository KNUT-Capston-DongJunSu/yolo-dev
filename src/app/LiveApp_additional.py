import queue
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from picamera import PiCamera
from picamera.array import PiRGBArray
from src.app.Density import DensityManager
from src.app.Pyplot import PlotManager
from src.services.email_transfer import EmailManager
from src.services.ftp_transfer import FTPmanager
from src.services.html_generator import HtmlManager
from src.services.database import DatabaseManager
from src.yolo_trainer import YOLOTrainer

class AdditionalApplicationHandler:
    def __init__(self, weight_path, db_config, email_config, ftp_config):
        # 카메라 및 모델 초기화
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 30
        self.rawCapture = PiRGBArray(self.camera, size=(640, 480))

        # PiCamera 설정
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # H264 코덱 사용
        self.video_writer = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (640, 480))  # MP4 포맷으로 저장

        # Model, Density 및 Pyplot 초기화
        self.model = YOLOTrainer(weight_path=weight_path)
        self.density_manager = DensityManager(self.frame_height)
        self.plot_manager = PlotManager()
        self.database_manager = DatabaseManager(**db_config)

        # 큐로 프레임 데이터 관리 (최대 5개)
        self.frame_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()

        self.email_manager = EmailManager(**email_config)
        self.ftp_manager = FTPmanager(**ftp_config)
        self.html_manager = HtmlManager()

    def camera_capture(self):
        """카메라에서 이미지를 캡처하여 큐에 추가 및 비디오 저장"""
        self.log_progress(
            process_name="camera_capture", step_name="capture per frames", 
            status="started", details="Initializing camera capture loop")

        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            if self.stop_event.is_set():
                break
            image = frame.array

            # 큐가 가득 찼을 경우 대기
            try:
                self.frame_queue.put(image, timeout=1)  # 큐에 이미지 추가
            except queue.Full:
                self.log_progress(
                    process_name="camera_capture",
                    step_name="queue.Full", 
                    status="error", 
                    details="Queue is full, skipping frame"
                    )

            self.rawCapture.truncate(0)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                self.stop_event.set()
                break

        self.log_progress(
            process_name="camera_capture", 
            step_name="capture per frames",
            status="completed", 
            details="Camera capture loop terminated"
            )

    def process_frames(self):
        """큐에서 프레임을 가져와 처리"""
        self.log_progress(
            process_name="process_frames", 
            step_name="frame processing",
            status="started", 
            details="Initializing frame processing loop"
            )

        while not self.stop_event.is_set():
            try:
                # 큐에서 프레임 가져오기
                image = self.frame_queue.get(timeout=1)

                # YOLO 모델로 예측 (예: 밀도 계산)
                # YOLO 모델로 예측
                results = self.model.predict(
                    frame=image,
                    conf_threshold=0.3,
                    iou_threshold=0.6,
                    save=False
                )
                density = self.density_manager.calculate_density(results["boxes"])
                plot_bgr = self.custom_plot(frame=image, results=results)
                
                # 결과 쓰기
                self.video_writer.write(plot_bgr)
                cv2.imshow("YOLO Stream", plot_bgr)

                # 밀도 값을 실시간 그래프에 업데이트
                self.plot_manager.update_Live_pyplot(density)

                self.log_progress(
                    process_name="process_frames", step_name="frame processed",
                    status="in progress", details=f"Density calculated: {density:.2f}", density=density)
            except queue.Empty:
                self.log_progress(process_name="process_frames",
                    step_name="queue.Empty", status="error", details="Queue is empty, waiting for frames...")
                time.sleep(0.1)
            except Exception as e:
                self.log_progress(process_name="process_frames",
                    step_name="processing_error", status="error", details=f"Unexpected error during frame processing: {e}")
                break

        self.log_progress(
            process_name="process_frames", step_name="frame processing",
            status="completed", details="Frame processing loop terminated")
    
    def app_start_running(self):
        """애플리케이션 실행"""
        self.log_progress(
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

            self.log_progress(
                process_name="Run", step_name="threads completed",
                status="completed", details="All threads completed successfully")
        except Exception as e:
            error_message = f"Error in thread execution: {e}"
            self.log_progress(process_name="Run",
                step_name="thread_error", status="error", details=error_message)
            self.email_manager.SendEmail(
                    subject="Error Notification: Application Processing", body=error_message)
        finally:
            self.stop_event.set()  # 모든 스레드 종료 신호
            self.camera.close()  # PiCamera 리소스 해제
            self.video_writer.release()  # VideoWriter 리소스 해제
            cv2.destroyAllWindows()  # OpenCV 윈도우 종료
            self.log_progress(
                process_name="Run", 
                step_name="application terminated",
                status="completed", 
                details="Application has stopped"
                )
            print("Application stopped.")

    def log_progress(self, step_name, status, details, density=None):
        """로깅 작업 통합"""
        self.database_manager.insert_progresslogs(
            process_name="process_frames", step_name=step_name, status=status, details=details)
        self.html_manager.append_html(
            process_name="process_frames", step_name=step_name, status=status, details=details, density=density)
        
    def custom_plot(self, frame, results):
        # OpenCV 이미지를 PIL 이미지로 변환
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)

        # 예측 결과에서 박스 가져오기
        boxes = results["prediction"]["boxes"]

        # 박스만 그리기
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)

        # PIL 이미지를 OpenCV 포맷으로 변환
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# 실행 예제
if __name__ == "__main__":
    db_config = {"host": "localhost", "user": "your_user", 
                 "password": "your_password", "database": "your_database"}
    email_config = {
        "sender_email": "your_email@gmail.com",
        "sender_password": "your_email_password", 
        "recipient_email": "recipient_email@gmail.com"
        }
    ftp_config = {
        "ftp_server":"ftp_server", 
        "ftp_user":"user_nmae", 
        "ftp_password":"password"
        }

    DensityAI_LiveApplication = ApplicationHandler(
                                weight_path="path/to/weights", 
                                db_config=db_config, email_config=email_config, ftp_config=ftp_config)
    DensityAI_LiveApplication.app_start_running()
