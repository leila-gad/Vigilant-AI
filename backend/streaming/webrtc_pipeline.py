import asyncio
import logging
import threading
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

try:
    from .data_channel import DataChannelHandler
except ImportError:
    from backend.streaming.data_channel import DataChannelHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc")

class WebRTCPipeline:
    def __init__(self, config: dict, camera_id: str):
        self.config = config
        self.camera_id = camera_id
        self.pc = None
        self.relay = None
        self.track = None
        self.data_handler = DataChannelHandler(None)
        self.running = False
        self.frame_queue = None
        self.loop = None
        self.thread = None

    async def create_peer_connection(self):
        self.pc = RTCPeerConnection()
        self.relay = MediaRelay()

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                self.track = self.relay.subscribe(track)

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            self.data_handler.channel = channel

    def build_pipeline(self):
        # aiortc gère le pipeline WebRTC automatiquement
        pass

    async def _run_webrtc(self):
        while self.running:
            try:
                frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
                if self.pc and self.pc.connectionState == "connected":
                    # Convertir le frame numpy en VideoFrame aiortc
                    video_frame = VideoFrame.from_ndarray(frame_data, format="bgr24")
                    # Ici on pourrait envoyer le frame via WebRTC
                    # Pour l'instant, on le stocke juste
                    pass
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"WebRTC error: {e}")

    def start(self):
        self.running = True
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.frame_queue = asyncio.Queue(maxsize=10)

        self.loop.run_until_complete(self.create_peer_connection())

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run_webrtc())

    def push_frame(self, frame: np.ndarray):
        if self.running and self.loop:
            try:
                self.loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.frame_queue.put(frame))
                )
            except Exception as e:
                logger.error(f"Failed to push frame: {e}")

    def update_quality(self, quality: dict):
        # aiortc gère automatiquement la qualité basée sur la connexion
        # On pourrait ajuster les paramètres ici si nécessaire
        pass

    async def _stop_async(self):
        self.running = False
        if self.pc:
            await self.pc.close()

    def stop(self):
        if self.loop and self.thread:
            self.loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._stop_async())
            )
            self.thread.join(timeout=2.0)