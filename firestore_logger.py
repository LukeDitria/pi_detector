from google.cloud import firestore
from google.cloud import storage
import datetime

class FirestoreLogger():
    def __init__(self, project_id, firestore_collection):
        self.project_id = project_id
        self.firestore_collection = firestore_collection

        self.db = firestore.Client(project=self.project_id)
        self.storage_client = storage.Client(project=self.project_id)

        data_dict = {"status": "on",
                     "timestamp": datetime.now()}
        self.log_data_to_firestore(data_dict,
                                   doc_type="startup")

    def log_data_to_firestore(self, data_dict, doc_type, timestamp=None):
        """Log data to Firestore."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]

        document_name = f"{doc_type}_{timestamp}"
        doc_ref = self.db.collection(self.firestore_collection).document(document_name)

        doc_ref.set(data_dict)
