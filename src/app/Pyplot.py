import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import deque
import cv2
import numpy as np
import sys  # sys 모듈 추가

class PlotManager:
    
    # 설정: 표시할 데이터의 최대 길이
    max_length = 100  # x축에 표시할 데이터 수
    
    def __init__(self, video_filename="results/graph_output.avi"):
        # 그래프 초기 설정
        self.video_fps = 10

        self.fig, self.ax = plt.subplots()
        self.x_data = deque(maxlen=self.max_length)  # 고정 길이 큐 (최대 max_length 개 유지)
        self.y_data = deque(maxlen=self.max_length)  # 고정 길이 큐 (최대 max_length 개 유지)
        self.line, = self.ax.plot([], [], label="Real-time Data")

        # x축 포맷 설정 (날짜 및 시간)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # 시간 형식
        self.fig.autofmt_xdate()  # x축 라벨 회전

        # 축 설정
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Density")
        self.ax.legend()

        # 동영상 설정
        self.video_filename = video_filename
        self.video_writer = None  # 동영상 writer 초기화
        print(f"Saving video to {self.video_filename}")

    @property
    def fps(self):
        return self.video_fps
    
    @fps.setter
    def fps(self, fps):
        self.video_fps = fps

    def init_video_writer(self):
        """VideoWriter 객체 초기화"""
        if self.video_writer is None:
            width, height = self.fig.canvas.get_width_height()
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 동영상 코덱
            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, self.video_fps, (width, height))

    def update_Live_pyplot(self, current_value, filename="graph.png"):
        current_time = datetime.now()  # 현재 시간
        
        # 최신 데이터를 큐에 추가 (최대 길이를 넘으면 자동 제거)
        self.x_data.append(current_time)
        self.y_data.append(current_value)
        print(f"current value: {current_value}")

        # 그래프 데이터 업데이트
        self.line.set_xdata(list(self.x_data))
        self.line.set_ydata(list(self.y_data))
        
        # x축 범위를 최근 max_length 데이터로 유지
        self.ax.set_xlim(self.x_data[0], self.x_data[-1])
        if len(self.y_data) > 1:  # 데이터가 있을 때만 y축 조정
            current_min = min(self.y_data)
            current_max = max(self.y_data)
            buffer = (current_max - current_min) * 0.1  # 최소값과 최대값의 10% 여유
            self.ax.set_ylim(current_min - buffer, current_max + buffer)

        # 그래프 업데이트
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        print(f"Graph complete to update")
    
        # 그래프를 동영상으로 저장
        self.init_video_writer()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV는 BGR 형식 사용
        self.video_writer.write(img)
        cv2.imshow("Density", img)
        print("Frame added to video.")

    def close(self):
        if self.video_writer:
            self.video_writer.release()
            print(f"Video saved as {self.video_filename}")
        plt.close('all')  # 모든 matplotlib 창 닫기
        cv2.destroyAllWindows()  # OpenCV 창 닫기
        sys.exit(0)  # 프로그램 강제 종료
        print("Graph window closed.")
    