================================================================
  Vigilant-AI — Intelligent Low-Latency Surveillance Gateway
  Backend Developer README
  INPT INE2 SmartICT Services IP et Multimédia — 2025-2026
================================================================

Vigilant-AI is a smart real-time video surveillance system.
It sits between your camera and your browser, using AI to
analyze video locally — without sending anything to the cloud.
The system streams processed video to a browser in under 200ms.

Think of it as a security guard that:
  - Watches camera feeds
  - Detects people and objects
  - Blurs faces for privacy
  - Streams everything live to a web dashboard

PROJECT STRUCTURE
vigilant-ai/
├── backend/      → Python AI brain + streaming (Person 1 + Person 2)
├── frontend/     → Web dashboard UI (Person 3)
└── models/       → YOLOv11 model file — downloaded, not written

backend/
[ test_camera.py ]
Run this first before anything else.
Opens your camera and shows the feed in a window.
  - Uses OpenCV to open the camera
  - Camera index 1 works on this machine
  - Press Q to quit

[ ai_detector.py ]
The core AI file. Loads YOLOv11 and detects up to 80 objects.
  - Loads yolov11n.pt from the models/ folder
  - detect(frame) returns list of objects with label, confidence, coords
  - draw_boxes(frame) draws rectangles around detected objects
  - Detects: person, car, phone, laptop, chair, dog, and 74 more

[ blur_faces.py ]
Privacy compliance module. Blurs all faces before video is sent.
  - Uses OpenCV built-in face detector — no extra download needed
  - apply(frame) blurs all faces in the frame
  - count_faces(frame) returns number of faces detected
  - The unmasked video never leaves the server — GDPR compliant

[ quality_switch.py ]
Adaptive streaming module. Saves bandwidth intelligently.
  - Default mode: 5 FPS at 480p (low bandwidth)
  - Person/vehicle detected: switches to 30 FPS at 1080p
  - After 5 seconds with no detection: switches back to low quality
  - get_current_mode() returns current FPS, resolution, mode name

[ main.py ]
The entry point. Connects all 3 AI files and runs everything.
  1. Loads ai_detector.py
  2. Loads blur_faces.py
  3. Loads quality_switch.py
  4. Opens the camera
  5. Runs a loop: detect → blur → switch quality → display
  6. Leaves TODO placeholders for Person 2 (GStreamer + WebRTC)


 HOW TO RUN
Step 1 — Install dependencies:
  pip install ultralytics opencv-python

Step 2 — Download the AI model:
  Download yolov11n.pt and place it in the models/ folder
  Link: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

Step 3 — Test your camera:
  python test_camera.py

Step 4 — Run the full system:
  python main.py


HOW THE DATA FLOWS
Step 1  Camera              Video frame is captured
Step 2  ai_detector.py      AI detects objects and returns bounding boxes
Step 3  blur_faces.py       All faces in the frame are blurred
Step 4  quality_switch.py   Quality is raised or lowered based on detections
Step 5  main.py             Processed frame shown and sent to GStreamer pipeline
Step 6  GStreamer pipeline   GStreamer encodes and streams via WebRTC
Step 7  browse frontend   Browser displays video with live bounding boxes

