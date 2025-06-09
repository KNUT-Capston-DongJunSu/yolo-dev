import cv2
import numpy as np
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
    
# 설정: 표시할 데이터의 최대 길이
max_length = 100  # x축에 표시할 데이터 수

# 그래프 초기 설정
video_fps = None

fig, ax = plt.subplots()
x_data = deque(maxlen=max_length)  # 고정 길이 큐 (최대 max_length 개 유지)
y_data = deque(maxlen=max_length)  # 고정 길이 큐 (최대 max_length 개 유지)
line, = ax.plot([], [], label="Real-time Data")

# x축 포맷 설정 (날짜 및 시간)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # 시간 형식
fig.autofmt_xdate()  # x축 라벨 회전

# 축 설정
ax.set_xlabel("Time")
ax.set_ylabel("Density")
ax.legend()

# 동영상 설정
video_filename = "results/predict/video/graph_output.mp4"
video_writer = None  # 동영상 writer 초기화

def initialize_plotter():
    global video_writer
    """VideoWriter 객체 초기화"""
    if video_writer is None:
        width, height = fig.canvas.get_width_height()
        video_writer = cv2.VideoWriter(
            video_filename, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            video_fps, 
            (width, height)
            )

def update_live_density(current_value):
    current_time = datetime.now()  # 현재 시간
    
    # 최신 데이터를 큐에 추가 (최대 길이를 넘으면 자동 제거)
    x_data.append(current_time)
    y_data.append(current_value)
    print(f"current value: {current_value}")

    # 그래프 데이터 업데이트
    line.set_xdata(list(x_data))
    line.set_ydata(list(y_data))
    
    # x축 범위를 최근 max_length 데이터로 유지
    ax.set_xlim(x_data[0], x_data[-1])
    if len(y_data) > 1:  # 데이터가 있을 때만 y축 조정
        current_min = min(y_data)
        current_max = max(y_data)
        buffer = (current_max - current_min) * 0.1  # 최소값과 최대값의 10% 여유
        ax.set_ylim(current_min - buffer, current_max + buffer)

    # 그래프 업데이트
    fig.canvas.draw()
    fig.canvas.flush_events()
    print(f"Graph complete to update")

    # 그래프를 동영상으로 저장
    initialize_plotter()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV는 BGR 형식 사용
    video_writer.write(img)
    cv2.imshow("Density", img)
    print("Frame added to video.")

def close_plotter():
    if video_writer:
        video_writer.release()
    plt.close('all')  # 모든 matplotlib 창 닫기
    cv2.destroyAllWindows()  # OpenCV 창 닫기
    
    