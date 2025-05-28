import queue
import threading
import cv2
import numpy as np
from PIL import Image, ImageDraw
# from picamera import PiCamera
# from picamera.array import PiRGBArray
from src.app.Density import DensityManager
from src.app.Pyplot import PlotManager
# from src.yolo_utils import YOLOTrainer

class ApplicationHandler:
    def __init__(self, weight_path):
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
        
        # 큐로 프레임 데이터 관리 (최대 5개)
        self.frame_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()

    def camera_capture(self):
        """카메라에서 이미지를 캡처하여 큐에 추가 및 비디오 저장"""
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            if self.stop_event.is_set():
                break
            image = frame.array

            # 큐가 가득 찼을 경우 대기
            try:
                self.frame_queue.put(image, timeout=1)  # 큐에 이미지 추가
            except queue.Full:
                pass

            self.rawCapture.truncate(0)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                self.stop_event.set()
                break

    def process_frames(self):
        """큐에서 프레임을 가져와 처리"""
        while not self.stop_event.is_set():
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
        except:
            pass
        finally:
            self.stop_event.set()  # 모든 스레드 종료 신호
            self.camera.close()  # PiCamera 리소스 해제
            self.video_writer.release()  # VideoWriter 리소스 해제
            cv2.destroyAllWindows()  # OpenCV 윈도우 종료
            print("Application stopped.")

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
    DensityAI_LiveApplication = ApplicationHandler(
        weight_path="path/to/weights"
        )
    DensityAI_LiveApplication.app_start_running()
