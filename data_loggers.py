from google.cloud import firestore
from google.cloud import storage
from datetime import datetime

import os
import utils
import logging
import json
import cv2

class DataLogger():
    def __init__(self, device_name, output_dir, save_images, log_remote, auto_select_media, firestore_project_id=None):

        self.device_name = device_name
        self.save_images = save_images

        self.log_remote = log_remote
        self.firestore_project_id = firestore_project_id

        if auto_select_media:
            self.data_output = os.path.join(utils.find_first_usb_drive(), "output")
        else:
            self.data_output = output_dir

        # Create output directories
        os.makedirs(self.data_output, exist_ok=True)
        self.image_detections_path = os.path.join(self.data_output, "images")
        os.makedirs(self.image_detections_path, exist_ok=True)

        self.json_detections_path = os.path.join(self.data_output, "detections")
        os.makedirs(self.json_detections_path, exist_ok=True)

        if self.log_remote:
            logging.info(f"Firestore remote logging")
            try:
                self.fire_logger = FirestoreLogger(project_id=self.firestore_project_id,
                                                   firestore_collection=self.device_name,
                                                   logger_type="data")
                logging.info(f"Firestore logging initialized")
            except Exception as e:
                logging.info(f"Firestore initialization failed: {e}")
                logging.info("Continuing without remote logging")
        else:
            self.fire_logger = None

    def log_data(self, detection_dict, frame, timestamp):

        timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S-%f")[:-3]

        # filename with timestamp with only the first 3 digits of the microseconds (milliseconds
        filename = f"{self.device_name}_{timestamp_str}"

        if self.save_images:
            # Save the frame locally
            lores_path = os.path.join(self.image_detections_path, f"{filename}.jpg")
            try:
                cv2.imwrite(lores_path, frame)
            except Exception as e:
                logging.info(f"Image saving failed: {e}")
        try:
            # Log detections locally
            json_path = os.path.join(self.json_detections_path, f"{filename}.json")
            with open(json_path, 'w') as f:
                json.dump(detection_dict, f, indent=2)

        except Exception as e:
            logging.info(f"Local detection logging failed: {e}")

        # Log detections to Firestore
        if self.log_remote and self.fire_logger:
            try:
                self.fire_logger.log_data_to_firestore(detection_dict,
                                                       doc_type="detection",
                                                       timestamp=timestamp,
                                                       add_time_to_dict=True)
            except Exception as e:
                logging.info(f"Firestore logging failed: {e}")


class FirestoreLogger():
    def __init__(self, project_id, firestore_collection, logger_type):
        self.project_id = project_id
        self.firestore_collection = firestore_collection

        self.db = firestore.Client(project=self.project_id)
        self.storage_client = storage.Client(project=self.project_id)

        logger_type_status = f"{logger_type}_startup"
        self.log_data_to_firestore({"status": "on"},
                                   doc_type=logger_type_status,
                                   add_time_to_dict=True)

    def log_data_to_firestore(self, data_dict, doc_type, timestamp=None, add_time_to_dict=False):
        """Log data to Firestore."""
        if timestamp is None:
            timestamp = datetime.now().astimezone()

        if add_time_to_dict:
            data_dict["timestamp"] = timestamp

        time_stamp_str = timestamp.strftime("%Y%m%d-%H%M%S-%f")[:-3]
        document_name = f"{doc_type}_{time_stamp_str}"
        doc_ref = self.db.collection(self.firestore_collection).document(document_name)

        doc_ref.set(data_dict)
