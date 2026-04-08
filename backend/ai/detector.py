import cv2
from ultralytics import YOLO
import numpy as np
from typing import List, Dict

class YOLODetector:
    def __init__(self, model_path: str, conf_thres: float = 0.5, iou_thres: float = 0.45):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                detections.append({"class": cls_name, "bbox": [x1, y1, x2, y2], "conf": conf})
        return detections