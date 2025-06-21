from picamera2.devices import Hailo
import logging
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, asdict
import utils
import numpy as np

import detectors.detector_utils as detector_utils
import utils


class HailoYolo:
    def __init__(self, model_path: str, labels_path: str, valid_classes_path: str, confidence: float):
        self.valid_classes_path = valid_classes_path
        self.confidence = confidence

        self.logger = logging.getLogger(__name__)

        self.yolo_model = Hailo(model_path)
        model_h, model_w, *_ = self.yolo_model.get_input_shape()

        self.model_wh = (model_w, model_h)
        self.hailo_aspect = (model_w / 640,
                             model_h / 640)

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

    def extract_detections(self, hailo_output) -> Optional[List[detector_utils.DetectionResultYOLO]]:
        """Extract detections from the HailoRT-postprocess output."""
        results = []
        for class_id, detections in enumerate(hailo_output):
            class_name = self.class_names[class_id]
            if self.valid_classes and class_name not in self.valid_classes:
                continue

            for detection in detections:
                score = float(detection[4])
                if score >= self.confidence:

                    y0, x0, y1, x1 = detection[:4]

                    box = {"xmin": float(x0) / self.hailo_aspect[0],
                           "ymin": float(y0) / self.hailo_aspect[1],
                           "xmax": float(x1) / self.hailo_aspect[0],
                           "ymax": float(y1) / self.hailo_aspect[1]}

                    detection = {"score": score,
                                 "class_name": class_name,
                                 "bbox": box}

                    results.append(detector_utils.DetectionResultYOLO.from_dict(detection))

        if len(results) > 0:
            return results
        else:
            return None

    def get_detections(self, frame: np.ndarray) -> Optional[List[detector_utils.DetectionResultYOLO]]:
        results = self.yolo_model.run(frame)

        # Extract and process detections
        detections = self.extract_detections(results)

        if detections:
            logging.info(f"Detected {len(detections)}")
            for detection in detections:
                class_name = detection.class_name
                score = detection.score
                logging.info(f"- {class_name} with confidence {score:.2f}")

        return detections



