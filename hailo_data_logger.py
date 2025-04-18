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
from data_loggers import DataLogger

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hailo object detection on camera stream")
    parser.add_argument("--config_file", type=str, help="Path to JSON configuration file")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save detection results")
    parser.add_argument("--start_delay", type=int, default=30,
                        help="Delay before running stream (default: 30)")
    parser.add_argument("--device_name", type=str, default="site1",
                        help="The name of this device to be used when saving data")

    parser.add_argument("--model", type=str, default="/usr/share/hailo-models/yolov8s_h8.hef",
                        help="Path for the HEF model")
    parser.add_argument("--labels", type=str, default="coco.txt",
                        help="Path to a text file containing labels")
    parser.add_argument("--valid_classes", type=str,
                        help="Path to text file containing list of valid class names to detect")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")

    parser.add_argument("--camera_type", type=str, default="csi", choices=["csi", "usb"],
                        help="What type of camera to use? csi/usb (default=csi)")
    parser.add_argument("--video_size", type=str, default="1920,1080",
                        help="Video size as width,height (default: 1920,1080)")
    parser.add_argument("--calibration_file", type=str, default="camera_calibration.pkl",
                        help="Camera calibration/correction parameters")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30)")
    parser.add_argument("--ips", type=int, default=5,
                        help="Inferences per second (default: 5)")
    parser.add_argument("--rotate_img", type=str, default="none",
                        help="Rotate/flip the input image: none, cw, ccw, flip", choices=["none", "cw", "ccw", "flip"])

    parser.add_argument("--project_id", type=str,
                        help="Google Cloud project ID")

    parser.add_argument("--buffer_secs", type=int, default=3,
                        help="The Circular buffer size in seconds (default: 3)")
    parser.add_argument("--detection_run", type=int, default=5,
                        help="Number of detections before recording (default: 5)")

    parser.add_argument("--log_remote", action='store_true', help="Log to remote store")
    parser.add_argument("--create_preview", action='store_true', help="Display the camera output")
    parser.add_argument("--save_video", action='store_true', help="Save video clips of detections")
    parser.add_argument("--save_images", action='store_true', help="Save images of the detections")
    parser.add_argument("--auto_select_media", action='store_true',
                        help="Auto selects a device mounted to /media to use as the storage device for outputs")
    parser.add_argument("--use_bgr", action='store_true',
                        help="Use BGR image not RGB")
    parser.add_argument("--crop_to_square", action='store_true',
                        help="Crop the input frame to be square")
    parser.add_argument("--convert_h264", action='store_true',
                        help="Convert the saved h264 video to mp4")
    parser.add_argument("--draw_bbox", action='store_true',
                        help="Draw bounding boxes on the saved images")
    return parser.parse_args()


class HailoLogger():
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
        logging.info("Capture Box Awake!")

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

        self.detector = HailoYolo(model_path=self.args.model, labels=self.args.labels,
                                  valid_classes=self.args.valid_classes, confidence=self.args.confidence)

        self.data_logger = DataLogger(device_name=self.args.device_name, output_dir=self.args.output_dir,
                                      save_images=self.args.save_images, log_remote=self.args.log_remote,
                                      auto_select_media=self.args.auto_select_media,
                                      firestore_project_id=self.args.project_id)

        # Parse video size
        if isinstance(self.args.video_size, str):
            self.video_w, self.video_h = map(int, self.args.video_size.split(','))
        else:
            # Handle case where video_size might be a list/tuple in the JSON
            self.video_w, self.video_h = self.args.video_size

        if self.args.camera_type == "csi":
            from csi_camera import CameraCSI
            self.camera = CameraCSI(device_name=self.args.device_name, video_wh=(self.video_w, self.video_h),
                                    model_wh=self.detector.model_wh, fps=self.args.fps, use_bgr=self.args.use_bgr,
                                    crop_to_square=self.args.crop_to_square, calibration_file=self.args.calibration_file,
                                    save_video=self.args.save_video, data_output=self.data_logger.data_output,
                                    buffer_secs=self.args.buffer_secs, create_preview=self.args.create_preview,
                                    rotate_img=self.args.rotate_img, convert_h264=self.args.convert_h264)
        elif self.args.camera_type == "usb":
            from usb_camera import CameraUSB
            self.camera = CameraUSB(device_name=self.args.device_name, video_wh=(self.video_w, self.video_h),
                                    model_wh=self.detector.model_wh, fps=self.args.fps, use_bgr=self.args.use_bgr,
                                    crop_to_square=self.args.crop_to_square, calibration_file=self.args.calibration_file,
                                    save_video=self.args.save_video, data_output=self.data_logger.data_output,
                                    buffer_secs=self.args.buffer_secs, create_preview=self.args.create_preview,
                                    rotate_img=self.args.rotate_img)

    def run_detection(self):
        detections_run = 0
        no_detections_run = 0
        encoding = False

        seconds_per_frame = 1/self.args.ips
        last_frame_time = time.time()

        logging.info("Wait for startup and battery monitor checks!")
        time.sleep(self.args.start_delay)
        logging.info("Starting!")
        try:
            while True:
                # Capture and process frame
                main_frame, frame = self.camera.get_frames()

                # Generate timestamp
                timestamp = datetime.now().astimezone()

                # Extract and process detections
                detections = self.detector.get_detections(frame)

                if detections:
                    detection_dict = self.detector.create_log_dict(detections)

                    detections_run += 1
                    no_detections_run = 0

                    if self.args.use_bgr:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    self.data_logger.log_data(detection_dict, frame, timestamp)

                else:
                    no_detections_run += 1
                    detections_run = 0

                # Trigger a video recoding event
                if self.args.save_video:
                    if detections_run == self.args.detection_run:
                        if not encoding:
                            self.camera.start_video_recording()
                            encoding = True
                    elif encoding and no_detections_run == self.args.buffer_secs * self.args.ips:
                            self.camera.stop_video_recording()
                            encoding = False

                # Maintain Inference Rate
                time_diff = time.time() - last_frame_time
                wait_time = max(0, seconds_per_frame - time_diff)
                time.sleep(wait_time)
                last_frame_time = time.time()

        except KeyboardInterrupt:
            logging.info("\nStopping capture...")

        finally:
            self.camera.stop_camera()

def main():
    logger = HailoLogger()
    logger.run_detection()

if __name__ == "__main__":
    main()