#!/usr/bin/env python3

import argparse
import cv2
import os
import time
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import Hailo
from google.cloud import firestore
from google.cloud import storage
from datetime import datetime
import utils


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hailo object detection on camera stream")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save detection results")
    parser.add_argument("--model", type=str, default="/usr/share/hailo-models/yolov8s_h8.hef",
                        help="Path for the HEF model")
    parser.add_argument("--labels", type=str, default="examples/hailo/coco.txt",
                        help="Path to a text file containing labels")
    parser.add_argument("--valid_classes", type=str,
                        help="Path to text file containing list of valid class names to detect")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--video_size", type=str, default="1280,640",
                        help="Video size as width,height (default: 1920,1080)")
    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second (default: 1)")
    parser.add_argument("--project_id", type=str, required=True,
                        help="Google Cloud project ID")
    parser.add_argument("--log_remote", action='store_true', help="Log to remote store")

    return parser.parse_args()


def initialize_cloud_clients(project_id):
    """Initialize Google Cloud clients."""
    db = firestore.Client(project=project_id)
    storage_client = storage.Client(project=project_id)
    return db, storage_client


def log_detection_to_firestore(db, filename, detections):
    """Log detection results to Firestore."""
    doc_ref = db.collection('detections').document(os.path.splitext(filename)[0])

    doc_data = {
        "timestamp": datetime.now(),
        "filename": filename,
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

def main():
    args = parse_arguments()
    time.sleep(10)

    # Initialize Google Cloud clients
    if args.log_remote:
        try:
            db, storage_client = initialize_cloud_clients(args.project_id)

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            doc_ref = db.collection('startup').document(timestamp)
            doc_data = {"startup": True}
            doc_ref.set(doc_data)
        except Exception as e:
            print(f"Firestore initialization failed: {e}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    image_detections_path = os.path.join(args.output_dir, "images")
    os.makedirs(image_detections_path, exist_ok=True)

    json_detections_path = os.path.join(args.output_dir, "detections")
    os.makedirs(json_detections_path, exist_ok=True)

    # Parse video size
    video_w, video_h = map(int, args.video_size.split(','))

    # Load class names and valid classes
    class_names = utils.read_class_list(args.labels)
    if args.valid_classes:
        valid_classes = utils.read_class_list(args.valid_classes)
        print(f"Monitoring for classes: {', '.join(sorted(valid_classes))}")
    else:
        valid_classes = None
        print(f"Monitoring all classes")

    # Initialize Hailo and camera
    with Hailo(args.model) as hailo:
        model_h, model_w, *_ = hailo.get_input_shape()
        print("Input Shape:", model_h, model_w)
        hailo_aspect = model_w / model_h

        with Picamera2() as picam2:
            # Configure camera streams
            main = {'size': (video_w, video_h), 'format': 'XRGB8888'}
            lores = {'size': (model_w, model_h), 'format': 'RGB888'}
            print("lores Shape:", lores['size'])
            controls = {'FrameRate': args.fps}

            config = picam2.create_still_configuration(main, lores=lores, controls=controls)
            picam2.configure(config)
            picam2.start()

            try:
                while True:
                    # Capture and process frame
                    (main_frame, frame), metadata = picam2.capture_arrays(["main", "lores"])

                    results = hailo.run(frame)

                    # Extract and process detections
                    detections = utils.extract_detections(results, class_names, valid_classes, args.confidence, hailo_aspect)

                    if detections:
                        # Generate filename with timestamp
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"{timestamp}.jpg"

                        # Save the frame locally
                        lores_path = os.path.join(image_detections_path, filename)
                        cv2.imwrite(lores_path, frame)

                        # Log detections locally
                        utils.log_detection(filename, json_detections_path, detections)

                        # Log detections to Firestore
                        if args.log_remote:
                            try:
                                log_detection_to_firestore(db, filename, detections)
                            except Exception as e:
                                print(f"Firestore logging failed: {e}")

                        print(f"Detected {len(detections)} objects in {filename}")
                        for class_name, _, score in detections:
                            print(f"- {class_name} with confidence {score:.2f}")

                    time.sleep(0.01)

            except KeyboardInterrupt:
                print("\nStopping capture...")

            finally:
                picam2.stop()


if __name__ == "__main__":
    main()