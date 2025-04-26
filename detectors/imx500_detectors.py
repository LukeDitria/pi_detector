from picamera2.devices import IMX500
from picamera2 import Metadata

import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, asdict
import utils
import numpy as np


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
        self.yolo_model.show_network_fw_progress_bar()
        model_w, model_h, *_ = self.yolo_model.get_input_size()

        self.model_wh = (model_w, model_h)

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

    def extract_detections(self, np_outputs: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract detections from the IMX500 output."""
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

        results = []
        for box, score, category in zip(boxes, scores, classes):
            class_name = self.class_names[int(category)]
            if self.valid_classes and class_name not in self.valid_classes:
                continue

            y0, x0, h, w = box
            bbox = (float(x0), float(y0), float(x0 + w), float(y0 + h))
            score = float(score)
            if score >= self.confidence:
                results.append(DetectionYOLO(class_name=class_name, bbox=bbox, score=score))

        if len(results) > 0:
            detection_dicts = [det.to_dict() for det in results]
            doc_data = {
                "type": "detections",
                "detections": detection_dicts
            }
            return doc_data
        else:
            return None


    def get_detections(self, metadata: Metadata) -> Optional[Dict[str, Any]]:
        results = self.yolo_model.get_outputs(metadata, add_batch=True)

        # Extract and process detections
        data_dict = self.extract_detections(results)

        if data_dict:
            detections = data_dict["detections"]
            logging.info(f"Detected {len(detections)}")
            for detection in detections:
                class_name = detection["class_name"]
                score = detection["score"]
                logging.info(f"- {class_name} with confidence {score:.2f}")

        return data_dict



