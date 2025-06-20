import time
import cv2
from datetime import datetime
import logging
import sys

from data_loggers import DataLogger
import get_args

class DetectorLogger:
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
        self.args = get_args.parse_arguments()

        if self.args.accel_device == "imx500" and not self.args.camera_type == "csi":
            ValueError("If you are using the IMX500 camera_type MUST be set to csi")

        if self.args.detector_type == "motion":
            from detectors.motion_detector import MotionDetector
            self.detector = MotionDetector(threshold=self.args.motion_threshold,
                                           motion_percent=self.args.motion_percent)
        elif self.args.detector_type == "yolo":
            if self.args.accel_device == "hailo":
                from detectors.hailo_detectors import HailoYolo
                self.detector = HailoYolo(model_path=self.args.model, labels_path=self.args.labels,
                                          valid_classes_path=self.args.valid_classes, confidence=self.args.confidence)
            elif self.args.accel_device == "imx500":
                from detectors.imx500_detectors import IMX500Yolo
                self.detector = IMX500Yolo(model_path=self.args.model, labels_path=self.args.labels,
                                          valid_classes_path=self.args.valid_classes, confidence=self.args.confidence)

        self.data_logger = DataLogger(device_name=self.args.device_name, output_dir=self.args.output_dir,
                                      save_data_local=self.args.save_data_local, save_images=self.args.save_images,
                                      draw_bbox=self.args.draw_bbox, log_remote=self.args.log_remote,
                                      auto_select_media=self.args.auto_select_media,
                                      firestore_project_id=self.args.project_id)

        # Parse video size
        if isinstance(self.args.video_size, str):
            self.video_w, self.video_h = map(int, self.args.video_size.split(','))
        else:
            # Handle case where video_size might be a list/tuple in the JSON
            self.video_w, self.video_h = self.args.video_size

        if self.args.camera_type == "csi":
            from cameras.csi_camera import CameraCSI
            self.camera = CameraCSI(device_name=self.args.device_name, video_wh=(self.video_w, self.video_h),
                                    model_wh=self.detector.model_wh, fps=self.args.fps, use_bgr=self.args.use_bgr,
                                    crop_to_square=self.args.crop_to_square, calibration_file=self.args.calibration_file,
                                    save_video=self.args.save_video, data_output=self.data_logger.data_output,
                                    buffer_secs=self.args.buffer_secs, create_preview=self.args.create_preview,
                                    rotate_img=self.args.rotate_img, convert_h264=self.args.convert_h264)
        elif self.args.camera_type == "usb":
            from cameras.usb_camera import CameraUSB
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
        last_log_time = time.time()

        if self.args.lps is not None:
            seconds_per_log = 1/self.args.lps
        else:
            seconds_per_log = 0

        logging.info("Wait for startup and battery monitor checks!")
        time.sleep(self.args.start_delay + 2)
        logging.info("Starting!")
        while True:
            # Generate timestamp
            timestamp = datetime.now().astimezone()

            # Capture and process frame
            main_frame, frame, metadata = self.camera.get_frames()
            if frame is None:
                continue

            if self.args.accel_device == "imx500":
                # Extract the detections for the IMX500 from the metadata
                data_list = self.detector.get_detections(metadata)
            else:
                # Process the frame and extract the detections
                data_list = self.detector.get_detections(frame)

            if data_list:
                detections_run += 1
                no_detections_run = 0

                # Log rate can be different to inference rate
                if time.time() - last_log_time >= seconds_per_log:
                    last_log_time = time.time()
                    self.data_logger.log_data(data_list, main_frame, timestamp)
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

def main():
    logger = DetectorLogger()
    logger.run_detection()

if __name__ == "__main__":
    main()