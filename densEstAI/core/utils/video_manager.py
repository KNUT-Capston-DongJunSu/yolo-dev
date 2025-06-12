import cv2

class BaseVideoWriter:
    def __init__(self):
        self._writer = None
        self._fps = 30

    @property
    def fps(self):
        return self._fps
    
    @fps.setter
    def fps(self, value):
        self._fps = value
    
    def init_writer(self, width, height, filename):
        if self._writer is None:
            self._writer = cv2.VideoWriter(
                filename, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                self._fps, (width, height)
                )
            
        return self._writer
    
    def write(self, frame):
        return self._writer.write(frame)
    
    def close_writer(self):
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows() 
            
class BaseVideoCap:
    def __init__(self):
        self._capture = None

    def init_cap(self, video_path):
        if self._capture is None:
            self._capture = cv2.VideoCapture(video_path)
            if not self._capture.isOpened():
                raise IOError(f"Cannot open video: {video_path}")
            fps = int(self._capture.get(cv2.CAP_PROP_FPS))
            frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) 
            frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return self._capture, fps, frame_width, frame_height
    
    def close_cap(self):
        self._capture.release()
        cv2.destroyAllWindows()