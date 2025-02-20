import cv2
import os
import time
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import Hailo
from google.cloud import firestore
from google.cloud import storage
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


def initialize_cloud_clients(project_id):
    """Initialize Google Cloud clients."""
    db = firestore.Client(project=project_id)
    storage_client = storage.Client(project=project_id)
    return db, storage_client


def upload_image_to_cloud_storage(storage_client, bucket_name, image_path, filename):
    """Upload image to Google Cloud Storage."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"detections/{filename}")
    blob.upload_from_filename(image_path)
    return blob.public_url


def log_detection_to_firestore(db, filename, detections, image_url):
    """Log detection results to Firestore."""
    doc_ref = db.collection('detections').document(os.path.splitext(filename)[0])

    doc_data = {
        "timestamp": datetime.now(),
        "filename": filename,
        "image_url": image_url,
        "detections": [
            {
                "class": class_name,
                "confidence": float(score),
                "bbox": {
                    "x0": float(bbox[0]),
                    "y0": float(bbox[1]),
                    "x1": float(bbox[2]),
                    "y1": float(bbox[3])
                }
            }
            for class_name, bbox, score in detections
        ]
    }

    doc_ref.set(doc_data)
