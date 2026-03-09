import cv2

class Masker:

    def __init__(self, blur_strength=30):
        print("[Masker] Loading face detector...")

        # OpenCV's built-in face detector (no extra download needed)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.blur_strength = blur_strength
        print("[Masker] Face detector ready.")


    def blur_region(self, frame, x, y, w, h):
        # Extract the region we want to blur
        region = frame[y:y+h, x:x+w]

        # Apply strong gaussian blur to that region
        blurred = cv2.GaussianBlur(
            region,
            (self.blur_strength * 2 + 1, self.blur_strength * 2 + 1),
            self.blur_strength
        )

        # Put the blurred region back into the frame
        frame[y:y+h, x:x+w] = blurred

        return frame


    def apply(self, frame, detections=[]):
        # ── Step 1: Detect faces using OpenCV ──────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # ── Step 2: Blur each face found ───────────────────
        for (x, y, w, h) in faces:
            frame = self.blur_region(frame, x, y, w, h)

        # ── Step 3: Also blur any 'face' from YOLO ─────────
        # In case YOLOv11 detected a face separately
        for d in detections:
            if d["label"] == "face":
                x1, y1, x2, y2 = d["coords"]
                w = x2 - x1
                h = y2 - y1
                frame = self.blur_region(frame, x1, y1, w, h)

        return frame


    def count_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return len(faces)


if __name__ == "__main__":

    import time

    CAMERA_INDEX = 1

    print("Starting live masking test...")
    print("Your face will be blurred in real time.")
    print("Press Q to quit.")

    # Load masker
    masker = Masker(blur_strength=30)

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

        # Count faces before blurring
        face_count = masker.count_faces(frame)

        # Apply blur to all faces
        frame = masker.apply(frame)

        # Show face count on screen
        cv2.putText(frame, f"Faces masked: {face_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Vigilant-AI — Privacy Masking", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test finished.")