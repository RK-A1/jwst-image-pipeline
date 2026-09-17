"""
Filesystem layout and constants shared by every stage.

All pipeline state lives under one data directory. Under Astro that has to be
include/data: the CLI mounts only dags/, plugins/, include/ and tests/ into the
containers, so anything written elsewhere disappears with the container. Setting
JWST_DATA_DIR points the pipeline somewhere else, which is how the tests get an
isolated warehouse.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FLICKR_USER = "nasawebbtelescope"

# Photos dated before launch are not sent for labelling, matching how the published
# dataset was built; all 3,268 legacy photos never labelled are dated before it. The
# cutoff is launch rather than first light (2022-07-12) because 10 kept observations
# are dated between the two, during commissioning. Photos with a missing or far-future
# date_taken are labelled: the date is uploader-supplied, and the Horsehead Nebula
# carries a date in 2124.
LAUNCH_DATE = "2021-12-25"

# DuckDB lets one process hold the file open for writing, and that writer locks out
# readers in other processes too. Every task that opens the warehouse runs in this
# one-slot Airflow pool, so the scheduler queues them instead of letting them collide.
WAREHOUSE_POOL = "duckdb_warehouse"


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def warehouse(self) -> Path:
        return self.root / "warehouse" / "jwst.duckdb"

    @property
    def images(self) -> Path:
        """Full-size originals as downloaded from Flickr, named <photo_id>.jpg."""
        return self.root / "images"

    @property
    def label_log(self) -> Path:
        """Append-only record of every model response. Tracked in git."""
        return self.root / "labels" / "raw_labels.jsonl"

    @property
    def dataset(self) -> Path:
        """The published dataset. Tracked in git, apart from the images."""
        return self.root / "dataset"

    @property
    def dataset_images(self) -> Path:
        return self.dataset / "images"

    @property
    def staging(self) -> Path:
        return self.root / "staging"

    @property
    def review(self) -> Path:
        return self.root / "review"

    @property
    def torch_cache(self) -> Path:
        return self.root / "torch_cache"


def paths() -> Paths:
    return Paths(Path(os.environ.get("JWST_DATA_DIR", PROJECT_ROOT / "include" / "data")))


def utcnow() -> datetime:
    """Naive UTC, which is what DuckDB TIMESTAMP columns hold."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def image_file_name(photo_id: str) -> str:
    return f"{photo_id}.jpg"
