import gi
import sys

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from pipelines.encode import get_encode_pipeline
from pipelines.webrtc import get_webrtc_pipeline


class VideoPipeline:

    def __init__(self):

        Gst.init(None)

        encode = get_encode_pipeline()
        webrtc = get_webrtc_pipeline()

        pipeline_description = f"""
        videotestsrc is-live=true
        ! {encode}
        ! {webrtc}
        """

        print("Pipeline description:")
        print(pipeline_description)

        self.pipeline = Gst.parse_launch(pipeline_description)

        self.loop = GLib.MainLoop()

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

    def start(self):

        print("Starting pipeline...")
        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("Interrupted")

        self.stop()

    def stop(self):

        print("Stopping pipeline...")
        self.pipeline.set_state(Gst.State.NULL)

    def on_message(self, bus, message):

        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print("ERROR:", err)
            print("Debug:", debug)
            self.stop()
            self.loop.quit()

        elif msg_type == Gst.MessageType.EOS:
            print("End of stream")
            self.stop()
            self.loop.quit()


def main():

    pipeline = VideoPipeline()
    pipeline.start()


if __name__ == "__main__":
    main()