import cv2
from .detector import detect_objects
from .blur import blur_sensitive_objects

def process(frame):

    results = detect_objects(frame)

    # 1. blur
    frame = blur_sensitive_objects(frame, results)

    events = []   # ✅ ici

    # 2. dessiner + créer events
    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # 🔹 ajouter event
        events.append({
            "object": cls,
            "confidence": conf,
            "coords": [x1, y1, x2, y2]
        })

        # 🔹 dessiner box
        label = f"{cls} {conf:.2f}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255))