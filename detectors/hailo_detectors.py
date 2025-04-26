from picamera2.devices import Hailo
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

    def extract_detections(self, hailo_output) -> Optional[Dict[str, Any]]:
        """Extract detections from the HailoRT-postprocess output."""
        results = []
        for class_id, detections in enumerate(hailo_output):
            class_name = self.class_names[class_id]
            if self.valid_classes and class_name not in self.valid_classes:
                continue

            for detection in detections:
                y0, x0, y1, x1 = detection[:4]
                bbox = (float(x0) / self.hailo_aspect[0], float(y0) / self.hailo_aspect[1],
                        float(x1) / self.hailo_aspect[0], float(y1) / self.hailo_aspect[1])
                score = float(detection[4])
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


    def get_detections(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        results = self.yolo_model.run(frame)

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



