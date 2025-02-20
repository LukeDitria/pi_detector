import cv2
import os
import time
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import Hailo
from datetime import datetime


def read_class_list(filepath):
    """Read list of class names from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def extract_detections(hailo_output, class_names, valid_classes, threshold=0.5, hailo_aspect=1):
    """Extract detections from the HailoRT-postprocess output."""
    results = []
    for class_id, detections in enumerate(hailo_output):
        class_name = class_names[class_id]
        if valid_classes and class_name not in valid_classes:
            continue

        for detection in detections:
            y0, x0, y1, x1 = detection[:4]
            bbox = (float(x0) / hailo_aspect, float(y0), float(x1) / hailo_aspect, float(y1))
            score = detection[4]
            if score >= threshold:
                results.append([class_name, bbox, score])
    return results


def log_detection(filename, output_dir, detections):
    """Log detection results to a JSON file."""
    import json
    from datetime import datetime

    timestamp = datetime.now().isoformat()
    results = {
        "timestamp": timestamp,
        "filename": filename,
        "detections": [
            {
                "class": class_name,
                "confidence": float(score),
                "bbox": bbox
            }
            for class_name, bbox, score in detections
        ]
    }

    json_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)