import cv2
import os
import numpy as np

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

def pre_process_image(image, rotate="cw", h=640, w=640):
    if rotate == "cw":
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == "ccw":
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate == "flip":
        image = cv2.rotate(image, cv2.ROTATE_180)

    # assuming that w > h
    h, w, _ = image.shape
    if not h == w:
        split = (w - h)//2
        image = np.ascontiguousarray(image[:, split:split+h])

    return image

def find_first_usb_drive():
    # Relies on raspi OS to auto mount USB storage to /media/username etc
    media_path = "/media"

    # Check if /media exists
    if not os.path.exists(media_path):
        return None

    # Get all user directories under /media (usually just one)
    media_items = os.listdir(media_path)

    for user_dir in media_items:
        user_path = os.path.join(media_path, user_dir)

        # If this is a directory
        if os.path.isdir(user_path):
            # Check for any subdirectories (mounted drives)
            try:
                usb_drives = os.listdir(user_path)
                if usb_drives:
                    # Return the first drive found
                    return os.path.join(user_path, usb_drives[0])
            except:
                pass

    # No USB drives found
    return None
