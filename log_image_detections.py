#!/usr/bin/env python3

import argparse
import cv2
import os
import time
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import Hailo

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
    return parser.parse_args()

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
            bbox = (float(x0)/hailo_aspect, float(y0), float(x1)/hailo_aspect, float(y1))
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

def main():
    args = parse_arguments()
    time.sleep(10)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    json_detections_path = os.path.join(args.output_dir, "detections")
    image_detections_path = os.path.join(args.output_dir, "images")
    os.makedirs(json_detections_path, exist_ok=True)
    os.makedirs(image_detections_path, exist_ok=True)

    # Parse video size
    video_w, video_h = map(int, args.video_size.split(','))

    # Load class names and valid classes
    class_names = read_class_list(args.labels)
    if args.valid_classes:
        valid_classes = read_class_list(args.valid_classes)
        print(f"Monitoring for classes: {', '.join(sorted(valid_classes))}")
    else:
        valid_classes = None
        print(f"Monitoring all classes")

    # Initialize Hailo and camera
    with Hailo(args.model) as hailo:
        model_h, model_w, *_ = hailo.get_input_shape()
        print("Input Shape:", model_h, model_w)
        hailo_aspect = model_w/model_h
        
        with Picamera2() as picam2:
            # Configure camera streams
            main = {'size': (video_w, video_h), 'format': 'XRGB8888'}
            # Configure lores to maintain aspect ratio

            # lores_w = int(round(model_w * (video_w/video_h)))
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
                    detections = extract_detections(results, class_names, valid_classes, args.confidence, hailo_aspect)
                    
                    if detections:
                        # Generate filename with timestamp
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"hailo-{timestamp}.jpg"

                        # Save the cropped and resized frame used for detection
                        lores_path = os.path.join(image_detections_path, filename)
                        cv2.imwrite(lores_path, frame)
                        
                        # Log detections
                        log_detection(filename, json_detections_path, detections)
                        
                        print(f"Detected {len(detections)} objects in {filename}")
                        for class_name, _, score in detections:
                            print(f"- {class_name} with confidence {score:.2f}")
                        
            except KeyboardInterrupt:
                print("\nStopping capture...")
            
            finally:
                picam2.stop()

if __name__ == "__main__":
    main()
