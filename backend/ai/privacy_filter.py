import cv2
import numpy as np
from typing import List, Dict

class PrivacyFilter:
    def __init__(self, blur_kernel_size: int = 51):
        self.blur_kernel = (blur_kernel_size, blur_kernel_size)

    def apply_masking(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        for det in detections:
            if det["class"] in ["face", "license-plate"]:
                x1, y1, x2, y2 = det["bbox"]
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred = cv2.GaussianBlur(roi, self.blur_kernel, 0)
                    frame[y1:y2, x1:x2] = blurred
        return frame