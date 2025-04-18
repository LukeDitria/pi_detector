import argparse
import os
import time
import cv2
import json
from datetime import datetime
import logging
import sys

import utils
from motion_detector import MotionDetector

def parse_arguments():
    parser = argparse.ArgumentParser(description="Motion detection on camera stream")
    parser.add_argument("--config_file", type=str, help="Path to JSON configuration file")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save detection results")
    parser.add_argument("--device_name", type=str, default="site1",
                        help="The name of this device to be used when saving data")

    parser.add_argument("--rotate_img", type=str, default="none",
                        help="Rotate/flip the input image: none, cw, ccw, flip", choices=["none", "cw", "ccw", "flip"])
    parser.add_argument("--threshold", type=int, default=25,
                        help="Pixel difference threshold (default: 25)")
    parser.add_argument("--motion_percent", type=float, default=0.25,
                        help="Percent of image to change to trigger event (default: 0.25)")
    parser.add_argument("--camera_type", type=str, default="csi", choices=["csi", "usb"],
                        help="What type of camera to use? csi/usb (default=csi)")

    parser.add_argument("--video_size", type=int, nargs='+', default=(1920, 1080),
                        help="Video size as width,height (default: 1920,1080)")
    parser.add_argument("--frame_size", type=int, nargs='+', default=(640, 480),
                        help="Width/Height of the image frame to be processed to detect motion (default: 640,480)")
    parser.add_argument("--calibration_file", type=str, default="camera_calibration.pkl",

                        help="Camera calibration/correction parameters")

    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30)")

    parser.add_argument("--ips", type=int, default=5,
                        help="Inferences per second (default: 5)")

    parser.add_argument("--project_id", type=str,
                        help="Google Cloud project ID")

    parser.add_argument("--firestore_collection", type=str, default="CameraBox",
                        help="This project name to be stored on Firestore")

    parser.add_argument("--buffer_secs", type=int, default=3,
                        help="The Circular buffer size in seconds (default: 3)")

    parser.add_argument("--detection_run", type=int, default=5,
                        help="Number of detections before recording (default: 5)")

    parser.add_argument("--start_delay", type=int, default=30,
                        help="Delay before running stream (default: 30)")

    parser.add_argument("--is_pi5", action='store_true', help="The devie a Raspberry Pi 5")
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
    return parser.parse_args()


class PixelMotionLogger():
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

        if self.args.save_video:
            self.videos_detections_path = os.path.join(self.data_output, "videos")
            os.makedirs(self.videos_detections_path, exist_ok=True)

        self.video_w, self.video_h = utils.parse_resolution(self.args.video_size)
        self.frame_w, self.frame_h = utils.parse_resolution(self.args.frame_size)

        self.detector = MotionDetector(threshold=self.args.threshold, motion_percent=self.args.motion_percent)

        if self.args.camera_type == "csi":
            from csi_camera import CameraCSI
            self.camera = CameraCSI(device_name=self.args.device_name, video_wh=(self.video_w, self.video_h),
                                    model_wh=(self.frame_w, self.frame_h),
                                    fps=self.args.fps, use_bgr=self.args.use_bgr, is_pi5=self.args.is_pi5,
                                    crop_to_square=self.args.crop_to_square,
                                    calibration_file=self.args.calibration_file, save_video=self.args.save_video,
                                    buffer_secs=self.args.buffer_secs, create_preview=self.args.create_preview,
                                    rotate_img=self.args.rotate_img, convert_h264=self.args.convert_h264)
        elif self.args.camera_type == "usb":
            from usb_camera import CameraUSB
            self.camera = CameraUSB(device_name=self.args.device_name, video_wh=(self.video_w, self.video_h),
                                    model_wh=(self.frame_w, self.frame_h), fps=self.args.fps, use_bgr=self.args.use_bgr,
                                    crop_to_square=self.args.crop_to_square,
                                    calibration_file=self.args.calibration_file, save_video=self.args.save_video,
                                    buffer_secs=self.args.buffer_secs, create_preview=self.args.create_preview,
                                    rotate_img=self.args.rotate_img)

    def initialize_cloud_clients(self):
        """Initialize Google Cloud clients."""
        from google.cloud import firestore
        from google.cloud import storage

        self.db = firestore.Client(project=self.args.project_id)
        self.storage_client = storage.Client(project=self.args.project_id)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        document_name = f"startup_{timestamp}"
        doc_ref = self.db.collection(self.args.firestore_collection).document(document_name)
        doc_data = {"type": "startup",
                    "timestamp": datetime.now()}
        doc_ref.set(doc_data)

    def log_detection_to_firestore(self, filename, detections):
        """Log detection results to Firestore."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        document_name = f"detection_{timestamp}"
        doc_ref = self.db.collection(self.args.firestore_collection).document(document_name)

        doc_data = {
            "type": "detections",
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

                # Extract and process detections
                detections = self.detector.get_detections(frame)

                if detections:
                    detections_run += 1
                    no_detections_run = 0

                    # Generate timestamp with only the first 3 digits of the microseconds (milliseconds)
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
                    filename = f"{self.args.device_name}_{timestamp}.jpg"

                    # Save the frame locally
                    if self.args.save_images:
                        lores_path = os.path.join(self.image_detections_path, filename)
                        if self.args.use_bgr:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        try:
                            cv2.imwrite(lores_path, frame)
                        except Exception as e:
                            logging.info(f"Image saving failed: {e}")

                    logging.info(f"Motion Detected!")
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
    logger = PixelMotionLogger()
    logger.run_detection()

if __name__ == "__main__":
    main()