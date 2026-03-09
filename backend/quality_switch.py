import time

class BitrateController:

    def __init__(self, default_fps=5, high_fps=30,
                 default_res="480p", high_res="1080p",
                 cooldown_seconds=5):

        self.default_fps = default_fps
        self.high_fps    = high_fps
        self.default_res = default_res
        self.high_res    = high_res
        self.cooldown    = cooldown_seconds

        # Start in low quality mode
        self.current_fps = default_fps
        self.current_res = default_res
        self.is_high_quality = False

        # Track when we last saw a person/vehicle
        self.last_detection_time = None

        print(f"[Bitrate] Started in low quality mode: {default_res} @ {default_fps} FPS")


    def update(self, detections):
        # Check if a person or vehicle was detected
        important_labels = ["person", "car", "truck", "motorcycle", "bus"]
        found_important = any(
            d["label"] in important_labels for d in detections
        )

        if found_important:
            # Update the last time we saw something important
            self.last_detection_time = time.time()

            # Switch to high quality if not already
            if not self.is_high_quality:
                self._switch_to_high()

        else:
            # Nothing detected — check if cooldown has passed
            if self.is_high_quality and self.last_detection_time:
                time_since_last = time.time() - self.last_detection_time

                if time_since_last >= self.cooldown:
                    # Cooldown passed → switch back to low quality
                    self._switch_to_low()


    def _switch_to_high(self):
        """Switch to high quality mode."""
        self.current_fps = self.high_fps
        self.current_res = self.high_res
        self.is_high_quality = True
        print(f"[Bitrate] ⬆ Switched to HIGH quality: {self.high_res} @ {self.high_fps} FPS — Person/Vehicle detected!")


    def _switch_to_low(self):
        """Switch back to low quality mode."""
        self.current_fps = self.default_fps
        self.current_res = self.default_res
        self.is_high_quality = False
        print(f"[Bitrate] ⬇ Switched to LOW quality: {self.default_res} @ {self.default_fps} FPS — Nothing detected.")


    def get_current_mode(self):
        """Returns current quality settings as a dictionary."""
        return {
            "fps":        self.current_fps,
            "resolution": self.current_res,
            "mode":       "HIGH" if self.is_high_quality else "LOW"
        }

if __name__ == "__main__":

    import cv2
    from ai_detector import Detector

    CAMERA_INDEX = 1
    MODEL_PATH   = "../models/yolov11n.pt"

    print("Starting bitrate controller test...")
    print("Stand in front of camera → quality switches to HIGH")
    print("Leave the frame → quality switches back to LOW after 5 seconds")
    print("Press Q to quit.")

    # Load detector and bitrate controller
    detector = Detector(model_path=MODEL_PATH, confidence=0.5)
    bitrate  = BitrateController(
        default_fps=5,
        high_fps=30,
        default_res="480p",
        high_res="1080p",
        cooldown_seconds=5
    )

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

        # Run detection
        detections = detector.detect(frame)

        # Update bitrate based on detections
        bitrate.update(detections)

        # Get current mode
        mode = bitrate.get_current_mode()

        # Draw detection boxes
        for d in detections:
            x1, y1, x2, y2 = d["coords"]
            color = (0, 255, 0) if d["label"] == "person" else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, d["label"], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show current quality mode on screen
        mode_color = (0, 255, 0) if mode["mode"] == "HIGH" else (0, 165, 255)
        cv2.putText(frame, f"Mode: {mode['mode']} | {mode['resolution']} @ {mode['fps']} FPS",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        # Show the frame
        cv2.imshow("Vigilant-AI — Bitrate Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test finished.")