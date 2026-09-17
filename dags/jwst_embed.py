"""
### jwst_embed

ResNet50 embeddings for downloaded photos that do not have one yet. Runs whenever
`jwst_ingest` reports new photos or images.

```
plan_batches ─▶ embed_batch ×N ─▶ store_embeddings ─▶ check_embeddings ─▶ publish_embeddings
               (mapped, no DB)     (one writer)
```

Each mapped `embed_batch` computes vectors for up to 32 images and writes them to a
staging file without touching the warehouse. `store_embeddings` then loads every
staged batch in one transaction. This is how the work runs in parallel even though
DuckDB accepts a single writer. It runs even if some batches failed, so their work is
kept, but the run is still marked failed so the failure gets noticed.

An image that cannot be decoded is recorded in `embedding_errors` and not retried
until the ingest downloads a fresh copy.
"""

from datetime import timedelta
from pathlib import Path

from airflow.exceptions import AirflowSkipException
from airflow.sdk import dag, get_current_context, task
from pendulum import datetime

from include.jwst_pipeline.assets import EMBEDDINGS, PHOTOS
from include.jwst_pipeline.config import WAREHOUSE_POOL


def _staging_dir() -> Path:
    from include.jwst_pipeline.config import paths

    run_id = get_current_context()["run_id"]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run_id)
    return paths().staging / "embeddings" / safe


@dag(
    schedule=PHOTOS,
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "jwst", "retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["jwst", "embeddings", "ml"],
    doc_md=__doc__,
)
def jwst_embed():

    @task(pool=WAREHOUSE_POOL)
    def plan_batches() -> list[list[list[str]]]:
        from include.jwst_pipeline import embeddings, warehouse

        with warehouse.session(read_only=True) as con:
            todo = embeddings.pending(con)
        size = embeddings.BATCH_SIZE
        return [[list(r) for r in todo[i:i + size]] for i in range(0, len(todo), size)]

    # Two at a time: each instance holds a ResNet50 and decodes 100-megapixel images.
    @task(max_active_tis_per_dagrun=2)
    def embed_batch(batch: list[list[str]]) -> str:
        from include.jwst_pipeline import embeddings
        from include.jwst_pipeline.config import paths

        dev = embeddings.device()
        model = embeddings.load_model(dev)
        items = [(photo_id, paths().images / image_file) for photo_id, image_file in batch]
        ids, vectors, errors = embeddings.embed(items, model, dev)

        index = get_current_context()["ti"].map_index
        out = _staging_dir() / f"batch-{index:05d}.npz"
        embeddings.save_batch(out, ids, vectors, errors)
        return str(out)

    @task(pool=WAREHOUSE_POOL, trigger_rule="all_done")
    def store_embeddings(batch_files: list[str | None]) -> int:
        from include.jwst_pipeline import embeddings, warehouse
        from include.jwst_pipeline.config import utcnow

        files = [Path(f) for f in (batch_files or []) if f]
        if not files:
            return 0
        with warehouse.session() as con:
            stored, failed = embeddings.store(con, files, now=utcnow())
        for f in files:
            f.unlink(missing_ok=True)
        return stored

    @task(pool=WAREHOUSE_POOL)
    def check_embeddings() -> None:
        from include.jwst_pipeline import quality, warehouse

        with warehouse.session(read_only=True) as con:
            quality.run(con, quality.EMBEDDING_CHECKS, context="embeddings")

    @task(outlets=[EMBEDDINGS])
    def publish_embeddings(stored: int) -> int:
        if not stored:
            raise AirflowSkipException("no new embeddings; dataset not rebuilt")
        return stored

    batches = embed_batch.expand(batch=plan_batches())
    stored = store_embeddings(batches)
    stored >> check_embeddings() >> publish_embeddings(stored)


jwst_embed()
