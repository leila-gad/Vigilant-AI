import cv2
import queue
import threading
import time
import numpy as np
from typing import Optional

class RTSPIngest:
    def __init__(self, rtsp_url: str, queue_size: int = 30):
        self.rtsp_url = rtsp_url
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.cap = None
        self.thread = None

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.thread = threading.Thread(target=self._ingest_loop, daemon=True)
        self.thread.start()

    def _ingest_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)  # auto-reconnect logic

    def get_frame(self) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=0.05)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.cap:
            self.cap.release()