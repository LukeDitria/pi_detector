import argparse
import os
import time
import cv2
import json
from datetime import datetime
import logging
import sys
import pickle

from picamera2 import Picamera2, Preview
from picamera2.devices import Hailo
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput

import utils
from hailo_yolo import HailoYolo

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hailo object detection on camera stream")
    parser.add_argument("--config_file", type=str, help="Path to JSON configuration file")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save detection results")
    parser.add_argument("--image_dir", type=str,
                        help="Directory of test images")
    parser.add_argument("--video_path", type=str,
                        help="Path to the test video")
    parser.add_argument("--model", type=str, default="/usr/share/hailo-models/yolov8s_h8.hef",
                        help="Path for the HEF model")
    parser.add_argument("--labels", type=str, default="coco.txt",
                        help="Path to a text file containing labels")
    parser.add_argument("--valid_classes", type=str,
                        help="Path to text file containing list of valid class names to detect")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--use_bgr", action='store_true',
                        help="Use BGR image not RGB")
    parser.add_argument("--crop_to_square", action='store_true',
                        help="Crop the input frame to be square")

    return parser.parse_args()


class HailoTester():
    def __init__(self):
        # Set up logging to stdout (systemd will handle redirection)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),  # Logs go to stdout (captured by systemd)
                logging.StreamHandler(sys.stderr)  # Warnings and errors go to stderr
            ]
        )
        logging.info("Hailo tester!")

        # First parse command line arguments with all defaults
        self.args = parse_arguments()

        # Load config file if provided and override CLI args
        if self.args.config_file:
            try:
                with open(self.args.config_file, 'r') as f:
                    config = json.load(f)

                # Override CLI args with JSON config values
                for key, value in config.items():
                    if hasattr(self.args, key):
                        setattr(self.args, key, value)

                logging.info(f"Loaded configuration from {self.args.config_file}")
            except Exception as e:
                logging.info(f"Error loading config file: {e}")
                logging.info("Using command line arguments instead")

        self.data_output = self.args.output_dir
        # Create output directories
        os.makedirs(self.data_output, exist_ok=True)
        self.image_detections_path = os.path.join(self.data_output, "images")
        os.makedirs(self.image_detections_path, exist_ok=True)

        # Load class names and valid classes
        self.class_names = utils.read_class_list(self.args.labels)
        if self.args.valid_classes:
            self.valid_classes = utils.read_class_list(self.args.valid_classes)
            logging.info(f"Monitoring for classes: {', '.join(sorted(self.valid_classes))}")
        else:
            self.valid_classes = None
            logging.info(f"Monitoring all classes")

        self.detector = HailoYolo(model_path=self.args.model, class_names=self.class_names,
                                  valid_classes=self.valid_classes, confidence=self.args.confidence)

        if self.args.image_dir:
            from media_processor import ImageProcessor
            self.image_source = ImageProcessor(path=self.args.image_dir, image_wh=self.detector.model_wh,
                                               crop_to_square=self.args.crop_to_square)
        elif self.args.video_path:
            from media_processor import VideoProcessor
            self.image_source = VideoProcessor(path=self.args.video_path, image_wh=self.detector.model_wh,
                                               crop_to_square=self.args.crop_to_square)
        else:
            ValueError("You must provide a video path or image dir!")

    def run_detection(self):

        logging.info("Starting!")
        frame_counter = 0
        for frame in self.image_source.get_frames():

            # Extract and process detections
            detections = self.detector.get_detections(frame)

            frame = utils.draw_detections(detections, frame)

            if detections:
                filename = f"{frame_counter}.jpg"

                lores_path = os.path.join(self.image_detections_path, filename)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(lores_path, frame)
                frame_counter += 1

def main():
    logger = HailoTester()
    logger.run_detection()

if __name__ == "__main__":
    main()