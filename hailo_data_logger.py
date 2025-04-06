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
    parser.add_argument("--model", type=str, default="/usr/share/hailo-models/yolov8s_h8.hef",
                        help="Path for the HEF model")
    parser.add_argument("--labels", type=str, default="coco.txt",
                        help="Path to a text file containing labels")
    parser.add_argument("--valid_classes", type=str,
                        help="Path to text file containing list of valid class names to detect")
    parser.add_argument("--rotate_img", type=str, default="none",
                        help="Rotate/flip the input image: none, cw, ccw, flip", choices=["none", "cw", "ccw", "flip"])
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
    parser.add_argument("--project_id", type=str,
                        help="Google Cloud project ID")
    parser.add_argument("--buffer_secs", type=int, default=3,
                        help="The Circular buffer size in seconds (default: 3)")
    parser.add_argument("--detection_run", type=int, default=5,
                        help="Number of detections before recording (default: 5)")
    parser.add_argument("--start_delay", type=int, default=30,
                        help="Delay before running stream (default: 30)")
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

        # Initialize Google Cloud clients
        if self.args.log_remote:
            if self.args.project_id is not None:
                try:
                    self.initialize_cloud_clients()
                except Exception as e:
                    logging.info(f"Firestore initialization failed: {e}")
            else:
                logging.info("You must provide a project ID to use Firestore!")

        if self.args.auto_select_media:
            self.data_output = os.path.join(utils.find_first_usb_drive(), "output")
        else:
            self.data_output = self.args.output_dir

        # Create output directories
        os.makedirs(self.data_output, exist_ok=True)
        self.image_detections_path = os.path.join(self.data_output, "images")
        os.makedirs(self.image_detections_path, exist_ok=True)

        self.json_detections_path = os.path.join(self.data_output, "detections")
        os.makedirs(self.json_detections_path, exist_ok=True)

        if self.args.save_video:
            self.videos_detections_path = os.path.join(self.data_output, "videos")
            os.makedirs(self.videos_detections_path, exist_ok=True)

        # Parse video size
        if isinstance(self.args.video_size, str):
            self.video_w, self.video_h = map(int, self.args.video_size.split(','))
        else:
            # Handle case where video_size might be a list/tuple in the JSON
            self.video_w, self.video_h = self.args.video_size

        # Load class names and valid classes
        self.class_names = utils.read_class_list(self.args.labels)
        if self.args.valid_classes:
            self.valid_classes = utils.read_class_list(self.args.valid_classes)
            logging.info(f"Monitoring for classes: {', '.join(sorted(self.valid_classes))}")
        else:
            self.valid_classes = None
            logging.info(f"Monitoring all classes")

        self.detector = HailoYolo(model_path=self.args.model, class_names=self.args.class_names,
                                  valid_classes=self.args.valid_classes, confidence=self.args.confidence)

        if self.args.camera_type == "csi":
            from csi_camera import CameraCSI
            self.camera = CameraCSI(video_wh=(self.video_w, self.video_h), model_wh=self.detector.model_wh,
                                    fps=self.args.fps, use_bgr=self.args.use_bgr, crop_to_square=self.args.crop_to_square,
                                    calibration_file=self.args.calibration_file, save_video=self.args.save_video,
                                    buffer_secs=self.args.buffer_secs, create_preview=self.args.create_preview,
                                    rotate_img=self.args.rotate_img)
        elif self.args.camera_type == "usb":
            from usb_camera import CameraUSB
            self.camera = CameraUSB(video_wh=(self.video_w, self.video_h), model_wh=self.detector.model_wh,
                                    fps=self.args.fps, use_bgr=self.args.use_bgr, crop_to_square=self.args.crop_to_square,
                                    calibration_file=self.args.calibration_file, save_video=self.args.save_video,
                                    buffer_secs=self.args.buffer_secs, create_preview=self.args.create_preview,
                                    rotate_img=self.args.rotate_img)

    def initialize_cloud_clients(self):
        """Initialize Google Cloud clients."""
        from google.cloud import firestore
        from google.cloud import storage

        self.db = firestore.Client(project=self.args.project_id)
        self.storage_client = storage.Client(project=self.args.project_id)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        doc_ref = self.db.collection('startup').document(timestamp)
        doc_data = {"startup": True}
        doc_ref.set(doc_data)

    def log_detection_to_firestore(self, filename, detections):
        """Log detection results to Firestore."""
        doc_ref = self.db.collection('detections').document(os.path.splitext(filename)[0])

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

                results = self.hailo.run(frame)

                # Extract and process detections
                detections = self.detector.get_detections(results)

                if detections:
                    detections_run += 1
                    no_detections_run = 0

                    # Generate timestamp with only the first 3 digits of the microseconds (milliseconds)
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
                    filename = f"{timestamp}.jpg"

                    # Save the frame locally
                    if self.args.save_images:
                        lores_path = os.path.join(self.image_detections_path, filename)
                        if self.args.use_bgr:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        try:
                            cv2.imwrite(lores_path, frame)
                        except Exception as e:
                            logging.info(f"Image saving failed: {e}")

                    try:
                        # Log detections locally
                        utils.log_detection(filename, self.json_detections_path, detections)
                    except Exception as e:
                        logging.info(f"Local detection logging failed: {e}")

                    # Log detections to Firestore
                    if self.args.log_remote:
                        try:
                            self.log_detection_to_firestore(filename, detections)
                        except Exception as e:
                            logging.info(f"Firestore logging failed: {e}")

                    logging.info(f"Detected {len(detections)} objects in {filename}")
                    for class_name, _, score in detections:
                        logging.info(f"- {class_name} with confidence {score:.2f}")
                else:
                    no_detections_run += 1
                    detections_run = 0

                # Trigger a video recoding event
                if self.args.save_video:
                    if detections_run == self.args.detection_run:
                        if not encoding:
                            self.camera.start_video_recording(self.videos_detections_path)
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