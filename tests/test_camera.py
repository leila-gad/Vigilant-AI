import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import numpy as np

Gst.init(None)

pipeline = Gst.parse_launch(
    "souphttpsrc location=http://192.168.57.58:8080/video ! "
    "decodebin ! videoconvert ! video/x-raw,format=BGR ! "
    "appsink name=sink"
)

appsink = pipeline.get_by_name("sink")

pipeline.set_state(Gst.State.PLAYING)

while True:
    sample = appsink.emit("pull-sample")
    buf = sample.get_buffer()
    caps = sample.get_caps()

    height = caps.get_structure(0).get_value('height')
    width = caps.get_structure(0).get_value('width')

    success, map_info = buf.map(Gst.MapFlags.READ)
    if not success:
        continue

    frame = np.frombuffer(map_info.data, np.uint8)
    frame = frame.reshape((height, width, 3))

    buf.unmap(map_info)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.set_state(Gst.State.NULL)
cv2.destroyAllWindows()