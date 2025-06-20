
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, asdict

import numpy as np

@dataclass
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def to_dict(self) -> dict:
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax
        }

@dataclass
class DetectionResult:
    score: float
    class_name: str
    bbox: BoundingBox

    @classmethod
    def from_dict(cls, detection_dict: dict) -> 'DetectionResult':
        return cls(
            score=detection_dict['score'],
            class_name=detection_dict['class_name'],
            bbox=BoundingBox(
                xmin=detection_dict['bbox']['xmin'],
                ymin=detection_dict['bbox']['ymin'],
                xmax=detection_dict['bbox']['xmax'],
                ymax=detection_dict['bbox']['ymax']
            )
        )

    def to_dict(self):
        result = {
            'score': self.score,
            'class_name': self.class_name,
            'bbox': self.bbox.to_dict()
        }
        return result

def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:

    # Calculate intersection coordinates
    x1 = max(box1.xmin, box2.xmin)
    y1 = max(box1.ymin, box2.ymin)
    x2 = min(box1.xmax, box2.xmax)
    y2 = min(box1.ymax, box2.ymax)

    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0

    # Calculate union area
    box1_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
    box2_area = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
    union = box1_area + box2_area - intersection

    return intersection / union

def apply_nms(detections: List[DetectionResult], nms_threshold: float = 0.3) -> List[DetectionResult]:
    """
    Apply Non-Maximum Suppression to filter overlapping detections.
    """
    if not detections:
        return []

    # Sort detections by confidence score
    detections = sorted(detections, key=lambda x: x.score, reverse=True)
    kept_detections = []

    while detections:
        # Keep the detection with highest confidence
        current = detections.pop(0)
        kept_detections.append(current)

        # Filter out detections with high IoU
        detections = [
            det for det in detections
            if compute_iou(current.box, det.box) < nms_threshold
        ]

    return kept_detections

