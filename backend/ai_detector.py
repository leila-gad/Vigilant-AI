import cv2
from ultralytics import YOLO


TARGETS = [
    # People
    "person",

    # Vehicles
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",

    # Outdoor
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",

    # Animals
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
    "bear", "zebra", "giraffe",

    # Accessories
    "backpack", "umbrella", "handbag", "tie", "suitcase",

    # Sports
    "frisbee", "skis", "snowboard", "sports ball", "kite", 
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",

    # Kitchen
    "bottle", "wine glass", "cup", "fork", "knife", 
    "spoon", "bowl",

    # Food
    "banana", "apple", "sandwich", "orange", "broccoli", 
    "carrot", "hot dog", "pizza", "donut", "cake",

    # Furniture
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",

    # Electronics
    "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",

    # Misc
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class Detector:

    def __init__(self, model_path, confidence=0.5):
        print(f"[Detector] Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        print("[Detector] Model loaded successfully.")


    def detect(self, frame):
        detections = []

        # Run the AI model on the frame
        results = self.model(frame, verbose=False)

        # Loop through everything the AI found
        for result in results:
            for box in result.boxes:

                # Get the label (what object is this?)
                label = self.model.names[int(box.cls)]

                # Get the confidence score
                confidence = float(box.conf)

                # Only keep detections we care about
                if label in TARGETS and confidence >= self.confidence:

                    # Get the bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    detections.append({
                        "label":      label,
                        "confidence": round(confidence, 2),
                        "coords":     [x1, y1, x2, y2]
                    })

        return detections


    def has_person(self, detections):
        """Returns True if a person was detected."""
        return any(d["label"] == "person" for d in detections)


    def has_vehicle(self, detections):
        """Returns True if a vehicle was detected."""
        vehicle_labels = ["car", "truck", "motorcycle", "bus"]
        return any(d["label"] in vehicle_labels for d in detections)


    def draw_boxes(self, frame, detections):
        for d in detections:
            x1, y1, x2, y2 = d["coords"]
            label = d["label"]
            confidence = d["confidence"]

            # Pick color based on label
            color = (0, 255, 0) if label == "person" else (255, 0, 0)

            # Draw the box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw the label text above the box
            text = f"{label} {int(confidence * 100)}%"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame


    def summary(self, detections):
        """Prints a simple summary of what was detected."""
        if not detections:
            print("[Detector] Nothing detected.")
            return

        counts = {}
        for d in detections:
            label = d["label"]
            counts[label] = counts.get(label, 0) + 1

        summary = ", ".join([f"{v}x {k}" for k, v in counts.items()])
        print(f"[Detector] Found: {summary}")

if __name__ == "__main__":

    import time

    # Your working camera index
    CAMERA_INDEX = 1

    # Model will auto-download if not present
    MODEL_PATH = "../models/yolov11n.pt"

    print("Starting live detection test...")
    print("Press Q to quit.")

    # Load detector
    detector = Detector(model_path=MODEL_PATH, confidence=0.5)

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(2)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Lost camera connection.")
            break

        # Run AI detection on the frame
        detections = detector.detect(frame)

        # Draw boxes on the frame
        frame = detector.draw_boxes(frame, detections)

        # Print what was found
        if detections:
            detector.summary(detections)

        # Show the frame
        cv2.imshow("Vigilant-AI — Live Detection", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test finished.")