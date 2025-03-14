import argparse
import sys
from functools import lru_cache

import cv2
import numpy as np
import time
import utils
import os

from picamera2 import MappedArray, Picamera2, Preview
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

class Imx500Logger():
    def __init__(self):
        self.args = self.get_args()

        # This must be called before instantiation of Picamera2
        self.imx500 = IMX500(self.args.model)
        self.intrinsics = self.imx500.network_intrinsics
        if not self.intrinsics:
            self.intrinsics = NetworkIntrinsics()
            self.intrinsics.task = "object detection"
        elif self.intrinsics.task != "object detection":
            print("Network is not an object detection task", file=sys.stderr)
            exit()

        # Override intrinsics from args
        for key, value in vars(self.args).items():
            if key == 'labels' and value is not None:
                with open(value, 'r') as f:
                    self.intrinsics.labels = f.read().splitlines()
            elif hasattr(self.intrinsics, key) and value is not None:
                setattr(self.intrinsics, key, value)

        # Defaults
        if self.intrinsics.labels is None:
            with open("coco_labels.txt", "r") as f:
                self.intrinsics.labels = f.read().splitlines()
        self.intrinsics.update_with_defaults()

        if self.args.print_intrinsics:
            print(self.intrinsics)
            exit()

        self.picam2 = Picamera2(self.imx500.camera_num)
        config = self.picam2.create_preview_configuration(controls={"FrameRate": self.intrinsics.inference_rate}, buffer_count=12)

        self.imx500.show_network_fw_progress_bar()
        self.picam2.configure(config)
        self.picam2.start_preview(Preview.QT)

        self.picam2.start()

        if self.intrinsics.preserve_aspect_ratio:
            self.imx500.set_auto_aspect_ratio()

        self.detections = None
        self.picam2.pre_callback = self.draw_detections

        # Create output directories
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.image_detections_path = os.path.join(self.args.output_dir, "images")
        os.makedirs(self.image_detections_path, exist_ok=True)

        self.json_detections_path = os.path.join(self.args.output_dir, "detections")
        os.makedirs(self.json_detections_path, exist_ok=True)

    def parse_detections(self, main, metadata):
        """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
        bbox_normalization = self.intrinsics.bbox_normalization
        bbox_order = self.intrinsics.bbox_order
        threshold = self.args.threshold
        iou = self.args.iou
        max_detections = self.args.max_detections

        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        input_w, input_h = self.imx500.get_input_size()
        if np_outputs is None:
            self.detections = None
            return None

        if self.intrinsics.postprocess == "nanodet":
            boxes, scores, classes = \
                postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                              max_out_dets=max_detections)[0]
            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            if bbox_normalization:
                boxes = boxes / input_h

            if bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]
            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)

        self.detections = [(category, box, score, self.imx500.convert_inference_coords(box, metadata, self.picam2))
            for box, score, category in zip(boxes, scores, classes)
            if score > threshold
        ]

        yolo_detections = [(category, box, score)
                           for box, score, category in zip(boxes, scores, classes)
                           if score > threshold
                           ]

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}.jpg"

        # Save the frame locally
        if self.args.save_images:
            lores_path = os.path.join(self.image_detections_path, filename)
            cv2.imwrite(lores_path, main)

        # Log detections locally
        utils.log_detection(filename, self.json_detections_path, yolo_detections)

    @lru_cache
    def get_labels(self):
        labels = self.intrinsics.labels

        if self.intrinsics.ignore_dash_labels:
            labels = [label for label in labels if label and label != "-"]
        return labels

    def draw_detections(self, request, stream="main"):
        """Draw the detections for this request onto the ISP output."""
        if self.detections is None:
            return

        labels = self.get_labels()
        with MappedArray(request, stream) as m:
            for detection in self.detections:
                x, y, w, h = detection[3]
                label = f"{labels[int(detection[0])]} ({detection[2]:.2f})"

                # Calculate text size and position
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = x + 5
                text_y = y + 15

                # Create a copy of the array to draw the background with opacity
                overlay = m.array.copy()

                # Draw the background rectangle on the overlay
                cv2.rectangle(overlay,
                              (text_x, text_y - text_height),
                              (text_x + text_width, text_y + baseline),
                              (255, 255, 255),  # Background color (white)
                              cv2.FILLED)

                alpha = 0.30
                cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

                # Draw text on top of the background
                cv2.putText(m.array, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Draw detection box
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)

            if self.intrinsics.preserve_aspect_ratio:
                b_x, b_y, b_w, b_h = self.imx500.get_roi_scaled(request)
                color = (255, 0, 0)  # red
                cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))

    def run_detector(self):
        while True:
            main, metadata = self.picam2.capture_arrays(["main", "lores"])
            self.parse_detections(main, metadata)

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, help="Path of the model",
                            default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
        parser.add_argument("--fps", type=int, help="Frames per second")
        parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
        parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                            help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
        parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
        parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
        parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
        parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
        parser.add_argument("--postprocess", choices=["", "nanodet"],
                            default=None, help="Run post process of type")
        parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                            help="preserve the pixel aspect ratio of the input tensor")
        parser.add_argument("--labels", type=str,
                            help="Path to the labels file")
        parser.add_argument("--output_dir", type=str, default="output",
                            help="Directory to save detection results")
        parser.add_argument("--print-intrinsics", action="store_true",
                            help="Print JSON network_intrinsics then exit")
        parser.add_argument("--save_images", action='store_true', help="Save images of the detections")

        return parser.parse_args()


if __name__ == "__main__":
    imx500_logger = Imx500Logger()
    imx500_logger.run_detector()