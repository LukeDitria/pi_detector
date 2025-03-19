import argparse
import os
import time
import cv2
import json
from datetime import datetime

from picamera2 import Picamera2, Preview
from picamera2.devices import Hailo
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput

import utils

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
                        help="Rotate/flip the input image: none, cw, ccw, flip")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--video_size", type=str, default="1920,1080",
                        help="Video size as width,height (default: 1920,1080)")
    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second (default: 1)")
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
    return parser.parse_args()


class HailoLogger():
    def __init__(self):
        # First parse command line arguments with all defaults
        self.args = parse_arguments()
        self.model_h, self.model_w = 640, 640

        # Load config file if provided and override CLI args
        if self.args.config_file:
            try:
                with open(self.args.config_file, 'r') as f:
                    config = json.load(f)

                # Override CLI args with JSON config values
                for key, value in config.items():
                    if hasattr(self.args, key):
                        setattr(self.args, key, value)

                print(f"Loaded configuration from {self.args.config_file}")
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using command line arguments instead")

        # Initialize Google Cloud clients
        if self.args.log_remote:
            if self.args.project_id is not None:
                try:
                    self.initialize_cloud_clients()
                except Exception as e:
                    print(f"Firestore initialization failed: {e}")
            else:
                print("You must provide a project ID to use Firestore!")

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
            print(f"Monitoring for classes: {', '.join(sorted(self.valid_classes))}")
        else:
            self.valid_classes = None
            print(f"Monitoring all classes")

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
        # Initialize Hailo and camera
        with Hailo(self.args.model) as hailo:
            self.model_h, self.model_w, *_ = hailo.get_input_shape()
            print("Input Shape:", self.model_h, self.model_w)
            self.hailo_aspect = self.model_w / self.model_h
            detections_run = 0
            no_detections_run = 0

            encoding = False

            with Picamera2() as picam2:
                # Configure camera streams
                main_res = {'size': (self.video_w, self.video_h), 'format': 'XRGB8888'}

                # Keep the aspect ratio of the main image in the lo-res image
                self.lores_w = int(round(self.model_w * (self.video_w / self.video_h)))

                if self.args.use_bgr:
                    lores_format = 'BGR888'
                else:
                    lores_format = 'RGB888'

                lores = {'size': (self.lores_w, self.model_h), 'format': lores_format}
                controls = {'FrameRate': self.args.fps}

                config = picam2.create_video_configuration(main_res, lores=lores, controls=controls)
                picam2.configure(config)

                if self.args.create_preview:
                    picam2.start_preview(Preview.QT, x=0, y=0, width=self.video_w, height=self.video_h)

                picam2.start()

                if self.args.save_video:
                    self.encoder = H264Encoder(1000000, repeat=True)
                    self.encoder.output = CircularOutput(buffersize=self.args.buffer_secs * self.args.fps)
                    picam2.start_encoder(self.encoder)
                    self.videos_detections_path = os.path.join(self.data_output, "videos")
                    os.makedirs(self.videos_detections_path, exist_ok=True)

                try:
                    while True:
                        # Capture and process frame
                        (main_frame, frame), metadata = picam2.capture_arrays(["main", "lores"])

                        # Resize and crop to model size
                        frame = utils.pre_process_image(frame, rotate=self.args.rotate_img,
                                                                   w=self.lores_w, h=self.model_h)
                        results = hailo.run(frame)

                        # Extract and process detections
                        detections = utils.extract_detections(results, self.class_names, self.valid_classes,
                                                              self.args.confidence, self.hailo_aspect)

                        if detections:
                            detections_run += 1
                            no_detections_run = 0

                            # Generate timestamp with only the first 3 digits of the microseconds (milliseconds)
                            timestamp = time.strftime("%Y%m%d-%H%M%S-%f")[:-3]
                            filename = f"{timestamp}.jpg"

                            # Save the frame locally
                            if self.args.save_images:
                                lores_path = os.path.join(self.image_detections_path, filename)
                                if self.args.use_bgr:
                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                                cv2.imwrite(lores_path, frame)

                            # Log detections locally
                            utils.log_detection(filename, self.json_detections_path, detections)

                            # Log detections to Firestore
                            if self.args.log_remote:
                                try:
                                    self.log_detection_to_firestore(filename, detections)
                                except Exception as e:
                                    print(f"Firestore logging failed: {e}")

                            print(f"Detected {len(detections)} objects in {filename}")
                            for class_name, _, score in detections:
                                print(f"- {class_name} with confidence {score:.2f}")
                        else:
                            no_detections_run += 1
                            detections_run = 0

                        if self.args.save_video:
                            if detections_run == self.args.detection_run:
                                if not encoding:
                                    epoch = int(time.time())
                                    file_name = os.path.join(self.videos_detections_path, f"{epoch}.h264")
                                    self.encoder.output.fileoutput = file_name
                                    self.encoder.output.start()
                                    encoding = True
                            elif encoding and no_detections_run == self.args.buffer_secs * self.args.fps:
                                    self.encoder.output.stop()
                                    encoding = False

                except KeyboardInterrupt:
                    print("\nStopping capture...")

                finally:
                    picam2.stop()

def main():
    logger = HailoLogger()
    logger.run_detection()

if __name__ == "__main__":
    main()