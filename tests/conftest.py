import os
import tempfile
from pathlib import Path

import pytest

# Keep Airflow from creating ~/airflow when the DAG tests import it on a host machine.
# Inside the Astro containers AIRFLOW_HOME is already set and this does nothing.
os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp(prefix="airflow-home-"))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    """Every test gets its own empty data directory, so nothing touches include/data."""
    root = tmp_path / "data"
    monkeypatch.setenv("JWST_DATA_DIR", str(root))
    return root


@pytest.fixture
def con(data_dir):
    from include.jwst_pipeline import warehouse

    c = warehouse.connect()
    warehouse.migrate(c)
    yield c
    c.close()


def make_image(path: Path, size=(64, 48), color=(200, 80, 40), fmt="JPEG") -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format=fmt)
    return path


def label_record(photo_id: str, **overrides) -> dict:
    record = {
        "photo_id": photo_id, "title": f"photo {photo_id}",
        "gate_category": "astronomical_observation", "subject": "nebula", "modality": "image",
        "object_name": None, "instrument": "unknown", "confidence": "high",
        "rationale": "test", "model": "claude-haiku-4-5", "backend": "api", "vision": True,
        "codebook_version": "v6", "seconds": 1.0, "prompt_tokens": 100,
        "output_tokens": 10, "cache_read_tokens": 0,
    }
    record.update(overrides)
    return record


@pytest.fixture
def add_photo(con, data_dir):
    """Insert a downloaded photo, writing a real image file for it."""
    from include.jwst_pipeline.config import utcnow

    def _add(photo_id: str, *, title: str | None = None, description: str = "",
             date_taken: str | None = "2023-01-01 00:00:00", with_file: bool = True,
             tags: list[str] | None = None, legacy_tag_label: str | None = None) -> str:
        image_file = bytes_ = None
        if with_file:
            path = make_image(data_dir / "images" / f"{photo_id}.jpg")
            image_file, bytes_ = path.name, path.stat().st_size
        con.execute("""
            INSERT INTO photos (photo_id, title, description, tags, date_taken, first_seen_at,
                                image_file, image_bytes, legacy_tag_label)
            VALUES (?, ?, ?, ?, CAST(? AS TIMESTAMP), ?, ?, ?, ?)
        """, [photo_id, title or f"photo {photo_id}", description, tags or [], date_taken,
              utcnow(), image_file, bytes_, legacy_tag_label])
        return photo_id

    return _add
