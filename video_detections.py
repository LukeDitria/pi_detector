#!/usr/bin/env python3

import argparse
import cv2
import os
import time
import numpy as np
from collections import deque
from picamera2 import Picamera2
from picamera2.devices import Hailo
from threading import Lock


class FrameBuffer:
    def __init__(self, buffer_size, frame_rate):
        """Initialize frame buffer with given size in seconds and frame rate."""
        self.max_frames = buffer_size * frame_rate
        self.buffer = deque(maxlen=self.max_frames)
        self.lock = Lock()

    def add_frame(self, frame):
        """Add a frame to the buffer thread-safely."""
        with self.lock:
            self.buffer.append(frame)

    def get_buffer_frames(self):
        """Get all frames from the buffer thread-safely."""
        with self.lock:
            return list(self.buffer)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hailo object detection on camera stream")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save detection results")
    parser.add_argument("--model", type=str, default="/usr/share/hailo-models/yolov8s_h8.hef",
                        help="Path for the HEF model")
    parser.add_argument("--labels", type=str, default="examples/hailo/coco.txt",
                        help="Path to text file containing labels")
    parser.add_argument("--valid_classes", type=str,
                        help="Path to text file containing list of valid class names to detect")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--video_size", type=str, default="1280,640",
                        help="Video size as width,height (default: 1920,1080)")
    parser.add_argument("--yolo_fps", type=int, default=5,
                        help="Frames per second to process the frames with the YOLO model (default: 5)")
    parser.add_argument("--video_fps", type=int, default=30,
                        help="Frames per second to stream the video (default: 30)")
    parser.add_argument("--num_dets", type=int, default=5,
                        help="Number of detections in a row to trigger recording (default: 5)")
    parser.add_argument("--prev_buffer_sec", type=int, default=1,
                        help="Number of seconds to keep in the buffer to save (default: 1)")

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

    json_path = os.path.join(output_dir, f"{filename}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


def create_video_writer(filename, fps, frame_size):
    """Create a VideoWriter object for saving video."""
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def main():
    args = parse_arguments()
    time.sleep(10)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    json_detections_path = os.path.join(args.output_dir, "detections")
    video_detections_path = os.path.join(args.output_dir, "videos")
    os.makedirs(json_detections_path, exist_ok=True)
    os.makedirs(video_detections_path, exist_ok=True)

    # Parse video size
    video_w, video_h = map(int, args.video_size.split(','))
    video_frame_size = (video_h, video_w)

    # Calculate frame skip for YOLO processing
    yolo_frame_interval = args.video_fps // args.yolo_fps
    frames_since_detection = 0

    # Initialize frame buffer
    frame_buffer = FrameBuffer(args.prev_buffer_sec, args.video_fps)

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
        hailo_aspect = model_w / model_h

        with Picamera2() as picam2:
            # Configure camera streams
            main = {'size': (video_w, video_h), 'format': 'XRGB8888'}
            lores = {'size': (model_w, model_h), 'format': 'RGB888'}

            print("lores Shape:", lores['size'])
            controls = {'FrameRate': args.video_fps}

            config = picam2.create_still_configuration(main, lores=lores, controls=controls)
            picam2.configure(config)
            picam2.start()

            try:
                frame_count = 0
                seq_detections = 0
                saving_video = False
                video_writer = None
                current_filename = None

                while True:
                    frame_count += 1

                    # Capture and process frame
                    (main_frame, lores_frame), metadata = picam2.capture_arrays(["main", "lores"])

                    # Write frame if we're currently saving
                    if saving_video and video_writer is not None:
                        video_writer.write(main_frame)

                    # Add frame to buffer
                    frame_buffer.add_frame(main_frame)

                    # Process detection at YOLO frame rate
                    if frame_count % yolo_frame_interval == 0:
                        results = hailo.run(lores_frame)
                        detections = extract_detections(results, class_names, valid_classes,
                                                        args.confidence, hailo_aspect)

                        if detections:
                            seq_detections += 1
                            frames_since_detection = 0

                            # Log detections
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            log_filename = f"hailo-{timestamp}"
                            log_detection(log_filename, json_detections_path, detections)

                            print(f"Detected {len(detections)} objects in {log_filename}")
                            for class_name, _, score in detections:
                                print(f"- {class_name} with confidence {score:.2f}")

                            if seq_detections == args.num_dets and not saving_video:
                                saving_video = True

                                # Generate filename with timestamp
                                timestamp = time.strftime("%Y%m%d-%H%M%S")
                                current_filename = f"hailo-{timestamp}"

                                # Create video writer
                                video_path = os.path.join(video_detections_path, f"{current_filename}.avi")
                                video_writer = create_video_writer(video_path, args.video_fps, video_frame_size)

                                # Write buffered frames
                                for buffered_frame in frame_buffer.get_buffer_frames():
                                    print("logging buffers")
                                    video_writer.write(buffered_frame)
                        else:
                            frames_since_detection += yolo_frame_interval
                            seq_detections = 0

                            if frames_since_detection >= args.video_fps:  # Wait 1 second without detections
                                frames_since_detection = 0
                                if saving_video:
                                    saving_video = False
                                    if video_writer is not None:
                                        print("release video_writer")
                                        video_writer.release()
                                        video_writer = None
                                    current_filename = None

            except KeyboardInterrupt:
                print("\nStopping capture...")

            finally:
                if video_writer is not None:
                    video_writer.release()
                picam2.stop()


if __name__ == "__main__":
    main()