from picamera2.devices import Hailo
import logging

class HailoYolo():
    def __init__(self, model_path, class_names, valid_classes, confidence):
        self.class_names = class_names
        self.valid_classes = valid_classes
        self.confidence = confidence

        self.logger = logging.getLogger(__name__)

        self.yolo_model = Hailo(model_path)
        model_h, model_w, *_ = self.yolo_model.get_input_shape()

        self.model_wh = (model_w, model_h)
        self.hailo_aspect = (model_w / 640,
                             model_h / 640)

        self.logger.info("Model initialized!")
        self.logger.info(f"Model input shape HxW: {model_h}, {model_w}")

    def extract_detections(self, hailo_output):
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
                score = detection[4]
                if score >= self.confidence:
                    results.append([class_name, bbox, score])

        return results

    def get_detections(self, frame):
        results = self.yolo_model.run(frame)

        # Extract and process detections
        detections = self.extract_detections(results)

        return detections



