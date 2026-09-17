"""
### jwst_ingest

Daily incremental ingest of the NASA Webb Flickr account into the warehouse.

```
migrate_warehouse ─▶ list_flickr_photos ─▶ record_sightings ─▶ fetch_metadata ─▶ upsert_photos
                                                                                     │
  publish_photos ◀─ check_photos ◀─ record_downloads ◀─ download_images ◀─ plan_downloads
```

**New photos** are found by comparing every id Flickr lists against the warehouse, not
by date, because Flickr's `date_taken` is supplied by the uploader and unreliable.

**Downloads** are planned from warehouse state, so any photo still without an image is
retried on the next run, up to three attempts. Files are written atomically and
decoded before they are accepted.

**Concurrency.** DuckDB accepts one writing process. Tasks that touch the warehouse run
in the one-slot `duckdb_warehouse` pool, and network calls happen in separate tasks,
so no task holds the pool while waiting on Flickr.

`publish_photos` emits the photos asset only when something changed, which is what
triggers `jwst_embed`.
"""

import os
from datetime import timedelta

from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.sdk import dag, task
from pendulum import datetime

from include.jwst_pipeline.assets import PHOTOS
from include.jwst_pipeline.config import WAREHOUSE_POOL


def _client():
    from include.jwst_pipeline.flickr import FlickrClient

    key = os.environ.get("FLICKR_API_KEY")
    if not key:
        raise AirflowFailException("FLICKR_API_KEY is not set; add it to .env")
    return FlickrClient(key)


@dag(
    schedule="@daily",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "jwst",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
        "retry_exponential_backoff": True,
    },
    tags=["jwst", "ingest", "flickr"],
    doc_md=__doc__,
)
def jwst_ingest():

    @task(pool=WAREHOUSE_POOL)
    def migrate_warehouse() -> list[int]:
        from include.jwst_pipeline import warehouse

        with warehouse.session() as con:
            return warehouse.migrate(con)

    @task
    def list_flickr_photos() -> list[str]:
        client = _client()
        return client.list_photo_ids(client.resolve_user_id())

    @task(pool=WAREHOUSE_POOL)
    def record_sightings(flickr_ids: list[str]) -> list[str]:
        from include.jwst_pipeline import ingest, warehouse

        with warehouse.session() as con:
            return ingest.record_sightings(con, flickr_ids)

    @task
    def fetch_metadata(new_ids: list[str]) -> list[dict]:
        from include.jwst_pipeline import ingest

        return ingest.fetch_metadata(_client(), new_ids) if new_ids else []

    @task(pool=WAREHOUSE_POOL)
    def upsert_photos(metadata: list[dict]) -> int:
        from include.jwst_pipeline import ingest, warehouse

        with warehouse.session() as con:
            return ingest.upsert_photos(con, metadata)

    @task(pool=WAREHOUSE_POOL)
    def plan_downloads() -> list[str]:
        from include.jwst_pipeline import ingest, warehouse

        with warehouse.session(read_only=True) as con:
            return ingest.pending_downloads(con)

    @task
    def download_images(photo_ids: list[str]) -> list[dict]:
        from include.jwst_pipeline import ingest

        return ingest.download_images(_client(), photo_ids) if photo_ids else []

    @task(pool=WAREHOUSE_POOL)
    def record_downloads(results: list[dict]) -> int:
        from include.jwst_pipeline import ingest, warehouse

        with warehouse.session() as con:
            ok, failed = ingest.record_downloads(con, results)
        if results and not ok:
            raise RuntimeError(f"all {failed} downloads failed")
        return ok

    @task(pool=WAREHOUSE_POOL)
    def check_photos() -> None:
        from include.jwst_pipeline import ingest, quality, warehouse

        checks = quality.PHOTO_CHECKS + [
            quality.Check("image files exist on disk", ingest.missing_image_files),
        ]
        with warehouse.session(read_only=True) as con:
            quality.run(con, checks, context="photos")

    @task(outlets=[PHOTOS])
    def publish_photos(inserted: int, downloaded: int) -> dict:
        if not inserted and not downloaded:
            raise AirflowSkipException("no new photos or images; downstream DAGs not triggered")
        return {"inserted": inserted, "downloaded": downloaded}

    migrated = migrate_warehouse()
    flickr_ids = list_flickr_photos()
    migrated >> flickr_ids

    inserted = upsert_photos(fetch_metadata(record_sightings(flickr_ids)))
    planned = plan_downloads()
    inserted >> planned

    downloaded = record_downloads(download_images(planned))
    checked = check_photos()
    downloaded >> checked >> publish_photos(inserted, downloaded)


jwst_ingest()
