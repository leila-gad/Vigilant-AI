import cv2
import threading
import time
import logging

logger = logging.getLogger("capture")

class GStreamerCapture:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.running = False
        self.thread = None

    def start(self):
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {self.rtsp_url}")

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started RTSP capture from {self.rtsp_url}")

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from RTSP stream")
                time.sleep(1)  # Retry delay
                continue
            # Frame processing would go here
            # For now, just log that we got a frame
            logger.debug(f"Captured frame: {frame.shape}")

    def read_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Stopped RTSP capture")