import numpy as np
from typing import Tuple, List, Dict
from .detector import YOLODetector
from .privacy_filter import PrivacyFilter
from .metadata_formatter import MetadataFormatter

class AIPipeline:
    def __init__(self, config: dict):
        ai_cfg = config["ai"]
        model_path = ai_cfg.get("model_path") or ai_cfg.get("weights")
        if not model_path:
            raise KeyError("Missing AI model path: 'ai.model_path' or 'ai.weights' required in config")
        confidence = ai_cfg.get("confidence_threshold") or ai_cfg.get("conf", 0.5)
        iou = ai_cfg.get("iou_threshold") or ai_cfg.get("iou", 0.45)
        self.detector = YOLODetector(model_path, confidence, iou)
        self.privacy = PrivacyFilter(config["privacy"]["blur_kernel"])
        self.formatter = MetadataFormatter()

    def process_frame(self, frame: np.ndarray, camera_id: str) -> Tuple[np.ndarray, str, List[Dict]]:
        detections = self.detector.detect(frame)
        processed = self.privacy.apply_masking(frame.copy(), detections)
        metadata = self.formatter.format(detections, camera_id)
        return processed, metadata, detections