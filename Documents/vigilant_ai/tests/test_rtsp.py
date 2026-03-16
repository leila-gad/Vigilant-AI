import cv2
from piipelines.capture import create_capture_pipeline
from piipelines.decode import get_frame
from bridge.ai_bridge import process_frame

url = "http://172.20.132.134:8080/video"

pipeline, appsink = create_capture_pipeline(url)

pipeline.set_state(1)

while True:

    frame = get_frame(appsink)

    frame = process_frame(frame)

    cv2.imshow("AI Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break