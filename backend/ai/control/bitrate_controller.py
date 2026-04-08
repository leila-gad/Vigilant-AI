from typing import Dict, List, Any

class BitrateController:
    def __init__(self, config: dict):
        self.config = config
        self.high_quality = False

    def decide_quality(self, detections: List[Dict]) -> Dict[str, Any]:
        important = {"person", "car", "vehicle", "face", "license-plate"}
        has_important = any(d["class"] in important for d in detections)
        if has_important and not self.high_quality:
            self.high_quality = True
            return {"fps": 30, "resolution": (1920, 1080), "bitrate": self.config["streaming"]["max_bitrate"], "level": "high"}
        elif not has_important and self.high_quality:
            self.high_quality = False
            return {"fps": 15, "resolution": (1280, 720), "bitrate": self.config["streaming"]["min_bitrate"], "level": "low"}
        return {}