import gi
import asyncio
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

class WebRTCPipeline:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.pipeline = None
        self.webrtc = None
        self.ai_bridge = None

    def set_ai_bridge(self, bridge):
        self.ai_bridge = bridge

    def build_pipeline(self):
        pipeline_str = f"""
        rtspsrc location={self.rtsp_url} latency=0 !
        rtph264depay !
        decodebin !
        videoconvert !
        tee name=t

        t. ! queue ! x264enc bitrate=2048 speed-preset=ultrafast tune=zerolatency !
        rtph264pay config-interval=1 pt=96 !
        application/x-rtp,media=video,encoding-name=H264,payload=96 !
        webrtcbin name=webrtc

        t. ! queue ! appsink name=appsink emit-signals=true
        """

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.webrtc = self.pipeline.get_by_name("webrtc")
        appsink = self.pipeline.get_by_name("appsink")

        appsink.connect("new-sample", self.on_new_sample)

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if self.ai_bridge:
            self.ai_bridge.process_sample(sample)
        return Gst.FlowReturn.OK

    async def start(self):
        self.build_pipeline()
        self.pipeline.set_state(Gst.State.PLAYING)

        print("Pipeline started")
        loop = asyncio.get_event_loop()
        await loop.run_forever()