import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import deque

class PlotManager:
    
    # 설정: 표시할 데이터의 최대 길이
    max_length = 100  # x축에 표시할 데이터 수
    
    def __init__(self):
        # 그래프 초기 설정
        plt.ion()  # interactive 모드 활성화
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

    def update_Live_pyplot(self, current_value, filename="FTP_files/graph.png"):
        current_time = datetime.now()  # 현재 시간
        
        # 최신 데이터를 큐에 추가 (최대 길이를 넘으면 자동 제거)
        self.x_data.append(current_time)
        self.y_data.append(current_value)

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
        self.fig.savefig(filename)
        print(f"Graph saved as {filename}")