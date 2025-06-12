import cv2
import numpy as np
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from densEstAI.core.utils.video_manager import BaseVideoWriter

# 동영상 설정
video_filename = "results/predict/video/graph_output.mp4"

class BaseDeque:
    # 설정: 표시할 데이터의 최대 길이
    max_length = 100  # x축에 표시할 데이터 수

    def __init__(self):
        self.data = deque(maxlen=BaseDeque.max_length)  # 고정 길이 큐 (최대 max_length 개 유지)
        
    def append(self, value):
        return self.data.append(value)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
class BasePlotter:
    def __init__(self, label='Live Density', xlabel='Time', ylabel='Density', fomat='%H:%M:%S'):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label=label)

        # x축 포맷 설정 (날짜 및 시간)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter(fomat))  # 시간 형식
        self.fig.autofmt_xdate()  # x축 라벨 회전

        # 축 설정
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()

    def update(self, current_time, current_value, x_data: BaseDeque, y_data: BaseDeque):
        # 최신 데이터를 큐에 추가 (최대 길이를 넘으면 자동 제거)
        x_data.append(current_time)
        y_data.append(current_value)
        print(f"current value: {current_value}")

        # 그래프 데이터 업데이트
        self.line.set_xdata(list(x_data))
        self.line.set_ydata(list(y_data))

        # x축 범위를 최근 max_length 데이터로 유지
        self.ax.set_xlim(x_data[0], x_data[-1])
        if len(y_data) > 1:
            y_min, y_max = min(y_data), max(y_data)
            buffer = (y_max - y_min) * 0.1 if y_max != y_min else 1
            self.ax.set_ylim(y_min - buffer, y_max + buffer)

        # 그래프 업데이트
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        print(f"Graph complete to update")


class LivePlotter(BasePlotter):
    def __init__(self):
        super().__init__()
        self.x_data = BaseDeque()
        self.y_data = BaseDeque()
        self.video_writer = BaseVideoWriter()
        width, height = self.fig.canvas.get_width_height()
        self.video_writer.init_writer(width, height, video_filename)
    
    def update_live_density(self, current_value):
        self.update(datetime.now(), current_value, self.x_data, self.y_data)
    
        img = self.convert_fig_to_frame(self.fig)
        self.video_writer.write(img)
        cv2.imshow("Density", img)
        print("Frame added to video.")

    @staticmethod
    def convert_fig_to_frame(fig):
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)