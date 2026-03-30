class DataChannel:
    def __init__(self, webrtc):
        self.channel = webrtc.emit("create-data-channel", "ai-data", None)

    def send(self, message):
        self.channel.emit("send-string", message)