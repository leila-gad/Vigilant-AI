import json
import time
from typing import List, Dict

class MetadataFormatter:
    @staticmethod
    def format(detections: List[Dict], camera_id: str) -> str:
        return json.dumps({
            "timestamp": time.time(),
            "camera_id": camera_id,
            "detections": detections
        })