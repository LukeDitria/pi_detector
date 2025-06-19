from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)
from picamera2 import Metadata

import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, asdict
import utils
import numpy as np
from libcamera import Rectangle, Size


@dataclass
class DetectionYOLO:
    class_name: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IMX500Yolo:
    def __init__(self, model_path: str, labels_path: str, valid_classes_path: str, confidence: float):
        self.valid_classes_path = valid_classes_path
        self.confidence = confidence

        self.logger = logging.getLogger(__name__)

        self.yolo_model = IMX500(model_path)
        self.intrinsics = self.yolo_model.network_intrinsics

        if not self.intrinsics:
            self.intrinsics = NetworkIntrinsics()
            self.intrinsics.task = "object detection"

        logging.info(f"postprocess: {self.intrinsics.postprocess}")

        self.yolo_model.show_network_fw_progress_bar()
        model_w, model_h = self.yolo_model.get_input_size()

        self.model_wh = (model_w, model_h)
        self.sensor_resolution = (4056, 3040)
        self.raw_resolution = (4056 // 2, 3040 // 2)

        # Load class names and valid classes
        self.class_names = utils.read_class_list(labels_path)
        if self.valid_classes_path:
            self.valid_classes = utils.read_class_list(self.valid_classes_path)
            logging.info(f"Monitoring for classes: {', '.join(sorted(self.valid_classes))}")
        else:
            self.valid_classes = None
            logging.info(f"Monitoring all classes")

        self.logger.info("Model initialized!")
        self.logger.info(f"Model input shape HxW: {model_h}, {model_w}")

    def get_scaled_obj(self, obj, isp_output_size, scaler_crop, sensor_output_size) -> Rectangle:
        """Scale the object coordinates based on the camera configuration and sensor properties."""
        # full_sensor = Rectangle(0, 0, 4056, 3040)
        # sensor_crop = scaler_crop.scaled_by(sensor_output_size, full_sensor.size)

        # obj_sensor = obj.scaled_by(sensor_output_size, full_sensor.size)
        # obj_bound = obj_sensor.bounded_to(sensor_crop)
        # obj_translated = obj_bound.translated_by(-sensor_crop.topLeft)
        # obj_scaled = obj_translated.scaled_by(isp_output_size, sensor_crop.size)

        obj_bound = obj.bounded_to(scaler_crop)
        obj_translated = obj_bound.translated_by(-scaler_crop.topLeft)
        obj_scaled = obj_translated.scaled_by(isp_output_size, scaler_crop.size)

        return obj_scaled

    def convert_inference_coords(self, coords: tuple, metadata: dict) -> tuple:
        """Convert relative inference coordinates into the output image coordinates space.
        The image passed to the YOLO model is a scaled version of the raw sensor data
        However, the output video stream from the camera is a crop and scaled version of the sensor image!
        So we need to do some funky transformations to make the model output line up with the video stream....
        This is mainly copied from picamera2/picamera2/devices/imx500/imx500.py
        """

        isp_output_size = Size(self.model_wh[0], self.model_wh[1])
        sensor_output_size = Size(self.raw_resolution[0], self.raw_resolution[1])
        scaler_crop = Rectangle(*metadata['ScalerCrop'])

        x0, y0, x1, y1 = coords
        full_sensor = Rectangle(0, 0, 4056, 3040)
        width, height = full_sensor.size.to_tuple()
        obj = Rectangle(
            *np.maximum(
                np.array([x0 * width, y0 * height, (x1 - x0) * width, (y1 - y0) * height]),
                0,
            ).astype(np.int32)
        )
        out = self.get_scaled_obj(obj, isp_output_size, scaler_crop, sensor_output_size)
        return out.to_tuple()

    def extract_detections(self, np_outputs: np.ndarray, metadata: dict) -> Optional[Dict[str, Any]]:
        """Extract detections from the IMX500 output."""
        if np_outputs:
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

            results = []
            for box, score, category in zip(boxes, scores, classes):
                class_name = self.class_names[int(category)]
                if self.valid_classes and class_name not in self.valid_classes:
                    continue

                x0, y0, x1, y1 = box

                bbox = (float(x0) / self.model_wh[0], float(y0) / self.model_wh[1],
                        float(x1) / self.model_wh[0], float(y1) / self.model_wh[1])

                bbox_xy_wh = self.convert_inference_coords(bbox, metadata)

                bbox_out = (bbox_xy_wh[0] / self.model_wh[0], bbox_xy_wh[1] / self.model_wh[1],
                            (bbox_xy_wh[0] + bbox_xy_wh[2]) / self.model_wh[0],
                            (bbox_xy_wh[1] + bbox_xy_wh[3]) / self.model_wh[1])

                score = float(score)
                if score >= self.confidence:
                    print(bbox_out)
                    results.append(DetectionYOLO(class_name=class_name, bbox=bbox_out, score=score))
                    logging.info(f"- {x0}, {y0}, {x1} {y1}: score {score}")

            if len(results) > 0:
                detection_dicts = [det.to_dict() for det in results]
                doc_data = {
                    "type": "detections",
                    "detections": detection_dicts
                }
                return doc_data
            else:
                return None
        else:
            return None

    def get_detections(self, metadata: Metadata) -> Optional[Dict[str, Any]]:
        results = self.yolo_model.get_outputs(metadata, add_batch=True)

        # Extract and process detections
        data_dict = self.extract_detections(results, metadata)

        if data_dict:
            detections = data_dict["detections"]
            logging.info(f"Detected {len(detections)}")
            for detection in detections:
                class_name = detection["class_name"]
                score = detection["score"]
                logging.info(f"- {class_name} with confidence {score:.2f}")

        return data_dict



