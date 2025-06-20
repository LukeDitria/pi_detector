from datetime import datetime
from typing import Union, Generator, List, Optional, Tuple, Dict, Any
import numpy as np

import os
import utils
import logging
import json
import cv2

class DataLogger:
    def __init__(self, device_name: str, output_dir: str, save_data_local: bool,
                 save_images: bool, draw_bbox: bool, log_remote: bool, auto_select_media: bool,
                 firestore_project_id: Optional[str]):

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Data Logger Created")

        self.device_name = device_name
        self.save_images = save_images
        self.draw_bbox = draw_bbox
        self.save_data_local = save_data_local

        self.logger.info(f"Saving Images locally: {str(self.save_images)}")
        self.logger.info(f"Saving detection data locally: {str(self.save_data_local)}")

        self.log_remote = log_remote
        self.firestore_project_id = firestore_project_id

        if auto_select_media:
            self.data_output = os.path.join(utils.find_first_usb_drive(), "output")
            if self.data_output is None:
                self.data_output = output_dir
                self.logger.warning(f"CANNOT find any media device! Defaulting to local saving!")
        else:
            self.data_output = output_dir

        # Create output directories
        os.makedirs(self.data_output, exist_ok=True)
        self.image_detections_path = os.path.join(self.data_output, "images")
        os.makedirs(self.image_detections_path, exist_ok=True)

        self.json_detections_path = os.path.join(self.data_output, "detections")
        os.makedirs(self.json_detections_path, exist_ok=True)

        if self.log_remote:
            self.logger.info(f"Firestore remote logging")
            try:
                self.fire_logger = FirestoreLogger(project_id=self.firestore_project_id,
                                                   firestore_collection=self.device_name,
                                                   logger_type="data")
                self.logger.info(f"Firestore logging initialized")
            except Exception as e:
                self.logger.info(f"Firestore initialization failed: {e}")
                self.logger.info("Continuing without remote logging")
                self.fire_logger = None
        else:
            self.fire_logger = None

    def log_data(self, detection_list: list, frame: np.ndarray, timestamp: datetime) -> None:

        timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S-%f")[:-3]

        # filename with timestamp with only the first 3 digits of the microseconds (milliseconds
        filename = f"{self.device_name}_{timestamp_str}"

        if self.save_images:
            if self.draw_bbox:
                try:
                    frame = utils.draw_detections(detection_list, frame)
                except Exception as e:
                    self.logger.info(f"Failed Drawing detections!: {e}")
            # Save the frame locally
            lores_path = os.path.join(self.image_detections_path, f"{filename}.jpg")
            try:
                cv2.imwrite(lores_path, frame)
            except Exception as e:
                self.logger.info(f"Image saving failed: {e}")

        if self.save_data_local:
            try:
                # Log detections locally
                json_path = os.path.join(self.json_detections_path, f"{filename}.json")
                with open(json_path, 'w') as f:
                    json.dump(detection_list, f, indent=2)

            except Exception as e:
                self.logger.info(f"Local detection logging failed: {e}")

        # Log detections to Firestore
        if self.log_remote and self.fire_logger:
            try:
                detection_dict = {"detections": detection_list}
                self.fire_logger.log_data_to_firestore(detection_dict,
                                                       doc_type="detection",
                                                       timestamp=timestamp,
                                                       add_time_to_dict=True)
            except Exception as e:
                self.logger.info(f"Firestore logging failed: {e}")


class FirestoreLogger():
    def __init__(self, project_id: str, firestore_collection: str, logger_type: str):
        self.project_id = project_id
        self.firestore_collection = firestore_collection
        from google.cloud import firestore
        from google.cloud import storage

        self.db = firestore.Client(project=self.project_id)
        self.storage_client = storage.Client(project=self.project_id)

        logger_type_status = f"{logger_type}_startup"
        self.log_data_to_firestore({"status": "on"},
                                   doc_type=logger_type_status,
                                   add_time_to_dict=True)

    def log_data_to_firestore(self, data_dict: dict, doc_type: str, timestamp: Optional[datetime] = None,
                              add_time_to_dict: bool = False) -> None:
        """Log data to Firestore."""
        if timestamp is None:
            timestamp = datetime.now().astimezone()

        if add_time_to_dict:
            data_dict["timestamp"] = timestamp

        time_stamp_str = timestamp.strftime("%Y%m%d-%H%M%S-%f")[:-3]
        document_name = f"{doc_type}_{time_stamp_str}"
        doc_ref = self.db.collection(self.firestore_collection).document(document_name)

        doc_ref.set(data_dict)
