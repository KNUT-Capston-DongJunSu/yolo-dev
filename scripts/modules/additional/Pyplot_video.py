import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Users\2R\ffmpeg\bin\ffmpeg.exe'
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.animation as animation
from collections import deque

class PlotManager:
    max_length = 100  # x축에 표시할 데이터 수

    def __init__(self, total_frames, output_video="output.mp4"):
        # 그래프 설정
        plt.ion()  # interactive 모드 활성화
        self.fig, self.ax = plt.subplots()
        self.x_data = deque(maxlen=self.max_length)
        self.y_data = deque(maxlen=self.max_length)
        # self.x_data = deque()  # maxlen 제거
        # self.y_data = deque()   
        self.line, = self.ax.plot([], [], label="Index Data")

        # # x축 포맷 설정 (날짜 및 시간)
        # self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # 시간 형식
        # self.fig.autofmt_xdate()  # x축 라벨 회전

        # 축 설정
        self.ax.set_xlabel("Index")
        self.ax.set_ylabel("Density")
        self.ax.legend()

        self.current_index = 0

        # 동영상 저장 설정
        self.output_video = output_video
        self.writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)

        # FuncAnimation 연결
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update_frame,
            frames=range(0, total_frames),
            blit=False,
            interval=100
        )

    def add_data(self, current_value):
        """새 데이터를 추가."""
        self.current_index += 1
        self.x_data.append(self.current_index)
        self.y_data.append(current_value)

        # 현재 추가된 데이터 디버깅 출력
        print(f"Added Data - Index: {self.current_index}, Value: {current_value}", flush=True)
        
    def update_frame(self, frame):
        """프레임 업데이트."""
        # 업데이트된 데이터를 선에 적용
        print(f"Drawing frame: {frame}")
        self.line.set_xdata(list(self.x_data))
        self.line.set_ydata(list(self.y_data))

        # x축 범위 업데이트
        self.ax.set_xlim(max(0, self.current_index - self.max_length), self.current_index)
        
        # y축 범위 업데이트
        if len(self.y_data) > 1:
            current_min = min(self.y_data)
            current_max = max(self.y_data)
            buffer = (current_max - current_min) * 0.1
            self.ax.set_ylim(current_min - buffer, current_max + buffer)
        # 그래프 업데이트
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return self.line,  # 업데이트된 그래프 선 반환
    
    def save_video(self):
        """애니메이션 동영상 저장"""
        with self.writer.saving(self.fig, self.output_video, dpi=100):
            self.ani.save(self.output_video, writer=self.writer)
        print(f"Video saved as {self.output_video}")


    # def save_png(self, filename="graph.png"):
    #     self.fig.savefig(filename)
    #     print(f"Graph saved as {filename}")

    # def save_gif(self, filename="real_time_graph.gif", fps=10):
    #     """GIF 저장."""
    #     writer = animation.PillowWriter(fps=fps)
    #     self.ani.save(filename, writer=writer)
    #     print(f"GIF saved as {filename}")

    # def save_video(self, filename="graph.mp4", fps=10):
    #     """동영상 저장."""
    #     writer = animation.FFMpegWriter(fps=fps)
    #     self.ani.save(filename, writer=writer)
    #     print(f"Video saved as {filename}")