import argparse
import logging
import json


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
    parser.add_argument("--detector_type", type=str, default="motion",
                        help="The type of detector to use", choices=["motion", "yolo"])
    parser.add_argument("--labels", type=str, default="coco.txt",
                        help="Path to a text file containing labels")
    parser.add_argument("--valid_classes", type=str,
                        help="Path to text file containing list of valid class names to detect")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")

    parser.add_argument("--motion_threshold", type=int, default=25,
                        help="Pixel difference threshold for pixel based motion detection (default: 25)")
    parser.add_argument("--motion_percent", type=float, default=0.25,
                        help="Percent of image to change to trigger event for pixel based motion detection (default: 0.25)")

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
    parser.add_argument("--save_data_local", action='store_true', help="Save detection data locally")

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

    parser.add_argument("--log_file_path", type=str, default="battery_logs.csv",
                        help="Directory to save detection results")
    parser.add_argument("--device_location", type=str, default="Melbourne",
                        help="Directory to save detection results")
    parser.add_argument("--log_rate_min", type=int, default=5,
                        help="Logging every (x) minutes (default: 5)")
    parser.add_argument("--shutdown_offset", type=int, default=0,
                        help="Offset (hours) to add from shutdown time pos/neg")
    parser.add_argument("--wakeup_offset", type=int, default=0,
                        help="Offset (hours) to add from wakeup time pos/neg")
    parser.add_argument("--operation_time", type=str,
                        help="When the device will operate: day, night, all", default='all',
                        choices=["day", "night", "all"])
    parser.add_argument("--low_battery_voltage", type=float, default=3.2,
                        help="Battery Voltage to shutdown at")
    parser.add_argument("--suptronics_ups", action='store_true',
                        help="Is a X1202 or X1206 UPS being used?")

    args = parser.parse_args()

    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                config = json.load(f)

            # Override CLI args with JSON config values
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)

            logging.info(f"Loaded configuration from {args.config_file}")
        except Exception as e:
            logging.info(f"Error loading config file: {e}")
            logging.info("Using command line arguments instead")

    return args
