import cv2
import time
import logging

from ai_detector import Detector
from blur_faces import Masker
from quality_switch import BitrateController

CONFIG = {
    "camera_index":         1,                      # Your working camera
    "model_path":           "../models/yolov11n.pt", # AI model path
    "confidence_threshold": 0.5,                    # AI confidence (50%)
    "default_fps":          5,                      # Low quality FPS
    "high_fps":             30,                     # High quality FPS
    "default_resolution":   "480p",                 # Low quality resolution
    "high_resolution":      "1080p",                # High quality resolution
    "cooldown_seconds":     5,                      # Seconds before low quality
    "blur_strength":        30,                     # Face blur strength
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("Vigilant-AI")


def main():
    log.info("Starting Vigilant-AI system...")

    # ── Step 1: Load AI Detector ───────────────────────────
    log.info("Loading AI detector...")
    detector = Detector(
        model_path=CONFIG["model_path"],
        confidence=CONFIG["confidence_threshold"]
    )
    log.info("AI detector ready.")

    # ── Step 2: Load Face Masker ───────────────────────────
    log.info("Loading face masker...")
    masker = Masker(
        blur_strength=CONFIG["blur_strength"]
    )
    log.info("Face masker ready.")

    # ── Step 3: Load Bitrate Controller ───────────────────
    log.info("Loading bitrate controller...")
    bitrate = BitrateController(
        default_fps=CONFIG["default_fps"],
        high_fps=CONFIG["high_fps"],
        default_res=CONFIG["default_resolution"],
        high_res=CONFIG["high_resolution"],
        cooldown_seconds=CONFIG["cooldown_seconds"]
    )
    log.info("Bitrate controller ready.")

    # ── Step 4: Open Camera ────────────────────────────────
    log.info(f"Opening camera index {CONFIG['camera_index']}...")
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(2)

    if not cap.isOpened():
        log.error("Could not open camera! Check your camera index.")
        return

    log.info("Camera ready.")

    # ── Step 5: Connect to GStreamer Pipeline ──────────────
    # TODO: Person 2 will plug in their pipeline here
    log.info("(Waiting for pipeline connection — Person 2)")

    # ── Step 6: Connect to WebRTC Signaling ───────────────
    # TODO: Person 2 will plug in their signaling here
    log.info("(Waiting for WebRTC connection — Person 2)")

    # ── Step 7: Main Loop ──────────────────────────────────
    log.info("System is running! Press Q to stop.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                log.error("Lost camera connection!")
                break

            # 1. Run AI detection on the frame
            detections = detector.detect(frame)

            # 2. Blur all faces for privacy
            frame = masker.apply(frame, detections)

            # 3. Update video quality based on detections
            bitrate.update(detections)

            # 4. Get current quality mode
            mode = bitrate.get_current_mode()

            # 5. Draw bounding boxes on detected objects
            for d in detections:
                x1, y1, x2, y2 = d["coords"]
                label = d["label"]
                confidence = d["confidence"]
                color = (0, 255, 0) if label == "person" else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {int(confidence*100)}%",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

            # 6. Show quality mode on screen
            mode_color = (0, 255, 0) if mode["mode"] == "HIGH" else (0, 165, 255)
            cv2.putText(frame,
                        f"Mode: {mode['mode']} | {mode['resolution']} @ {mode['fps']} FPS",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

            # 7. Show faces masked count
            face_count = masker.count_faces(frame)
            cv2.putText(frame, f"Faces masked: {face_count}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 8. Show the frame
            cv2.imshow("Vigilant-AI", frame)

            # 9. TODO: Send frame to Person 2's pipeline
            # pipeline.send_frame(frame, detections)

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        log.info("Shutting down...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        log.info("Cleanup complete. Goodbye!")


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    main()