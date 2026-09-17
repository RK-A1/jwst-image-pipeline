"""
Airflow assets connecting the DAGs. The only module in the package that imports
Airflow; the DAGs use these to schedule on each other's output instead of on clocks.

    jwst_ingest ──PHOTOS──▶ jwst_embed ──EMBEDDINGS──┐
                                                    ├──▶ jwst_dataset ──DATASET
    jwst_label  ──LABEL_LOG─────────────────────────┘
"""

from airflow.sdk import Asset

PHOTOS = Asset("jwst_warehouse_photos")
EMBEDDINGS = Asset("jwst_warehouse_embeddings")
LABEL_LOG = Asset("jwst_label_log")
DATASET = Asset("jwst_dataset")
