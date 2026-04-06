import numpy as np
from gi.repository import Gst

def get_frame(appsink):

    sample = appsink.emit("pull-sample")

    buf = sample.get_buffer()
    caps = sample.get_caps()

    height = caps.get_structure(0).get_value("height")
    width = caps.get_structure(0).get_value("width")

    success, map_info = buf.map(Gst.MapFlags.READ)

    frame = np.frombuffer(map_info.data, np.uint8)
    frame = frame.reshape((height, width, 3))

    buf.unmap(map_info)

    return frame