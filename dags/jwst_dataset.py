"""
### jwst_dataset

Rebuild the golden dataset. Runs when new labels or new embeddings arrive, or on demand.

```
load_labels ─▶ build_dataset ─▶ write_images ─▶ audit_dataset ─▶ publish_dataset ─▶ build_review_sheet
```

**Write, audit, publish.** The dataset is built into a staging directory and checked
there: unique keys, codebook values, one primary per release group, kept + rejected
accounting for every label, and the row count not falling below the published
build. Only a staging build that passes every check is swapped into
`include/data/dataset/`. A failed audit leaves the published dataset exactly as it was.

`load_labels` rebuilds the warehouse `labels` table from the committed label log and
refuses any record whose values are outside the codebook. An invalid gate value would
otherwise look like a rejection and silently drop a real observation.

Set `allow_shrink` to publish a build with fewer rows than the last one, after a
deliberate relabel, for example.
"""

from datetime import timedelta
from pathlib import Path

from airflow.sdk import Param, dag, get_current_context, task
from pendulum import datetime

from include.jwst_pipeline.assets import DATASET, EMBEDDINGS, LABEL_LOG
from include.jwst_pipeline.config import WAREHOUSE_POOL


@dag(
    schedule=(LABEL_LOG | EMBEDDINGS),
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "jwst", "retries": 1, "retry_delay": timedelta(minutes=1)},
    params={
        "allow_shrink": Param(False, type="boolean",
                              description="Publish even if the dataset has fewer rows than the last build."),
        "review_sample": Param(100, type="integer", minimum=0, maximum=1000,
                               description="Rows in the review sheet. 0 skips it."),
    },
    tags=["jwst", "dataset"],
    doc_md=__doc__,
)
def jwst_dataset():

    @task(pool=WAREHOUSE_POOL)
    def load_labels() -> int:
        from include.jwst_pipeline import labelling, quality, warehouse
        from include.jwst_pipeline.config import paths

        with warehouse.session() as con:
            warehouse.migrate(con)
            n = labelling.load_log(con, paths().label_log)
            quality.run(con, quality.LABEL_CHECKS, context="labels")
        return n

    @task(pool=WAREHOUSE_POOL)
    def build_dataset() -> str:
        from include.jwst_pipeline import assemble, warehouse
        from include.jwst_pipeline.config import paths

        run_id = get_current_context()["run_id"]
        staged = paths().staging / "dataset" / "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in run_id)
        with warehouse.session(read_only=True) as con:
            assemble.build(con, staged)
        return str(staged)

    @task(pool=WAREHOUSE_POOL)
    def write_images(staged: str) -> str:
        """Downscaled copies are keyed by photo_id and only ever added, so they are
        written straight into the dataset directory rather than staged."""
        import logging

        from include.jwst_pipeline import assemble, warehouse

        with warehouse.session(read_only=True) as con:
            written, present, failed = assemble.write_images(con)
            # Rebuild so image_file reflects the copies that now exist.
            if written:
                assemble.build(con, Path(staged))
        logging.getLogger(__name__).info("images: %d written, %d present, %d failed",
                                         written, present, failed)
        return staged

    @task
    def audit_dataset(staged: str) -> str:
        from include.jwst_pipeline import assemble

        allow_shrink = get_current_context()["params"]["allow_shrink"]
        assemble.audit(Path(staged), allow_shrink=allow_shrink)
        return staged

    @task(pool=WAREHOUSE_POOL, outlets=[DATASET])
    def publish_dataset(staged: str) -> dict:
        import json

        from include.jwst_pipeline import assemble, warehouse
        from include.jwst_pipeline.config import paths

        assemble.publish(Path(staged))
        with warehouse.session(read_only=True) as con:
            assemble.prune_images(con)
        return json.loads((paths().dataset / "manifest.json").read_text())["rows"]

    @task
    def build_review_sheet() -> str | None:
        from include.jwst_pipeline import review

        n = get_current_context()["params"]["review_sample"]
        return str(review.build_sheet(n)) if n else None

    built = build_dataset()
    load_labels() >> built
    publish_dataset(audit_dataset(write_images(built))) >> build_review_sheet()


jwst_dataset()
