import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

def create_capture_pipeline(url):

    pipeline = Gst.parse_launch(
        f"souphttpsrc location={url} ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink"
    )

    appsink = pipeline.get_by_name("sink")

    return pipeline, appsink