from enum import Enum

class CloudProvider(Enum):
    GCP = ("my-sh-project-398715", "predict_data", "prediction")

    def __init__(self, project_id: str, dataset_id: str, table_id: str) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
