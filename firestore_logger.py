from google.cloud import firestore
from google.cloud import storage
from datetime import datetime

class FirestoreLogger():
    def __init__(self, project_id, firestore_collection):
        self.project_id = project_id
        self.firestore_collection = firestore_collection

        self.db = firestore.Client(project=self.project_id)
        self.storage_client = storage.Client(project=self.project_id)

        self.log_data_to_firestore({"status": "on"}, doc_type="startup", add_time_to_dict=True)

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
