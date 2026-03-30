import cv2

def blur_sensitive_objects(frame, results):

    for box in results.boxes:

        cls = int(box.cls[0])   # classe
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # classes importantes (person, car, etc.)
        # YOLO class 0 = person
        if cls == 0:   # PERSON

            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                blurred = cv2.GaussianBlur(roi, (51, 51), 0)
                frame[y1:y2, x1:x2] = blurred

    return frame