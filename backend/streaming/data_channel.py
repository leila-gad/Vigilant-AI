class DataChannelHandler:
    def __init__(self, webrtcbin):
        self.channel = None  # populated by webrtcbin "on-data-channel" signal

    def send_metadata(self, json_str: str):
        if self.channel:
            # GstWebRTC.DataChannel.send_string(json_str) – real impl uses GObject signal
            pass  # placeholder; wired in webrtc_pipeline