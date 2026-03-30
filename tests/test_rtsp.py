import cv2
from pipelines.capture import create_capture_pipeline
from pipelines.decode import get_frame
from bridge.ai_bridge import process_frame

url = "http://192.168.57.58:8080/video"

pipeline, appsink = create_capture_pipeline(url)

pipeline.set_state(1)

while True:

    frame = get_frame(appsink)

    frame = process_frame(frame)

    cv2.imshow("AI Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break