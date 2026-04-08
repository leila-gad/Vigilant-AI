import threading
import queue
import time
from ..ingestion.rtsp_ingest import RTSPIngest
from ..ai.ai_pipeline import AIPipeline
from ..ai.control.bitrate_controller import BitrateController
from ..streaming.webrtc_pipeline import WebRTCPipeline

class AIBridge:
    def __init__(self, config: dict, camera_config: dict):
        self.config = config
        self.camera_id = camera_config.get("id", "camera")
        rtsp_url = camera_config.get("rtsp_url") or camera_config.get("url")
        if not rtsp_url:
            raise ValueError("Camera configuration must include 'rtsp_url' or 'url'")
        self.ingest = RTSPIngest(rtsp_url, config["ingestion"]["queue_size"])
        self.ai = AIPipeline(config)
        self.controller = BitrateController(config)
        self.webrtc = WebRTCPipeline(config, self.camera_id)
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.ingest.start()
        self.webrtc.start()
        self.thread = threading.Thread(target=self._bridge_loop, daemon=True)
        self.thread.start()

    def _bridge_loop(self):
        while self.running:
            frame = self.ingest.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            processed, metadata, detections = self.ai.process_frame(frame, self.camera_id)
            quality = self.controller.decide_quality(detections)
            if quality:
                self.webrtc.update_quality(quality)
            self.webrtc.push_frame(processed)
            self.webrtc.data_handler.send_metadata(metadata)  # DataChannel

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.ingest.stop()
        self.webrtc.stop()