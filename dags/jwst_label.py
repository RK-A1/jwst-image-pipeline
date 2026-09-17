"""
### jwst_label

Label unlabelled photos with Claude, or with a local model through Ollama.

**Manual trigger only.** Every other DAG runs on its own; this one spends money, so a
person decides when to run it, and `max_photos` caps the bill for each run. At about
half a cent per photo with Claude Haiku 4.5, the default cap of 100 costs about $0.50.

```
check_backend ─▶ plan_queue ─▶ label_photos ─▶ publish_label_log
```

The queue is every downloaded photo with no label and a `date_taken` not before
launch. Responses are appended to `include/data/labels/raw_labels.jsonl` as they
arrive, and photos already in that log are skipped. An interrupted run can therefore
be re-triggered without paying twice for any photo. Emitting the label-log asset
triggers `jwst_dataset`, which validates the new records and rebuilds the dataset.
"""

import os
import time
from datetime import timedelta

from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.sdk import Param, dag, get_current_context, task
from pendulum import datetime

from include.jwst_pipeline.assets import LABEL_LOG
from include.jwst_pipeline.config import WAREHOUSE_POOL

# Stop the run once this share of calls has failed, over at least this many attempts:
# failures on that scale mean a bad key, an exhausted balance or an outage, and
# carrying on would only make more failed calls.
MAX_FAILURE_RATE = 0.5
MIN_ATTEMPTS_BEFORE_ABORT = 5


@dag(
    schedule=None,
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "jwst", "retries": 1, "retry_delay": timedelta(minutes=5)},
    params={
        "max_photos": Param(100, type="integer", minimum=1, maximum=5000,
                            description="Most photos to label in this run: the spend cap."),
        "backend": Param("api", enum=["api", "ollama"]),
        "model": Param("", type="string",
                       description="Blank for the default: claude-haiku-4-5, or gemma4:12b on Ollama."),
        "vision": Param(True, type="boolean",
                        description="Send the image as well as the caption. Needed to catch "
                                    "'Artist's Concept' printed on the image."),
    },
    tags=["jwst", "labelling", "llm"],
    doc_md=__doc__,
)
def jwst_label():

    @task(retries=0)
    def check_backend() -> dict:
        from include.jwst_pipeline import labelling

        params = get_current_context()["params"]
        backend = params["backend"]
        if backend == "api" and not os.environ.get("ANTHROPIC_API_KEY"):
            raise AirflowFailException("ANTHROPIC_API_KEY is not set; add it to .env")
        default = labelling.DEFAULT_API_MODEL if backend == "api" else labelling.DEFAULT_OLLAMA_MODEL
        return {"backend": backend, "model": params["model"] or default,
                "vision": params["vision"], "max_photos": params["max_photos"]}

    @task(pool=WAREHOUSE_POOL)
    def plan_queue(settings: dict) -> list[list[str | None]]:
        from include.jwst_pipeline import labelling, warehouse
        from include.jwst_pipeline.config import paths

        with warehouse.session(read_only=True) as con:
            rows = labelling.queue(con, paths().label_log, settings["max_photos"])
        return [list(r) for r in rows]

    @task
    def label_photos(settings: dict, todo: list[list[str | None]]) -> int:
        import logging

        from include.jwst_pipeline import labelling
        from include.jwst_pipeline.config import paths

        log = logging.getLogger(__name__)
        if not todo:
            return 0

        log_path = paths().label_log
        done = {r["photo_id"] for r in labelling.read_log(log_path)}
        backend, model, vision = settings["backend"], settings["model"], settings["vision"]
        client = None
        if backend == "api":
            import anthropic

            client = anthropic.Anthropic()

        appended = failed = 0
        for i, (photo_id, title, desc, image_file) in enumerate(todo, 1):
            if photo_id in done:
                continue
            image_b64 = labelling.encode_image(paths().images / image_file) if vision else None
            started = time.monotonic()
            try:
                if backend == "api":
                    label, usage = labelling.call_api(client, model, title, desc, image_b64)
                else:
                    label, usage = labelling.call_ollama(model, title, desc, image_b64)
            except Exception as exc:
                failed += 1
                log.warning("[%d/%d] %s failed: %s", i, len(todo), photo_id, exc)
                attempts = appended + failed
                if attempts >= MIN_ATTEMPTS_BEFORE_ABORT and failed / attempts >= MAX_FAILURE_RATE:
                    raise RuntimeError(f"{failed} of {attempts} calls failed; stopping") from exc
                continue

            record = labelling.build_record(
                photo_id, title, label, usage, model=model, backend=backend,
                vision=vision, seconds=time.monotonic() - started,
            )
            labelling.append_log(log_path, record)
            appended += 1
            log.info("[%d/%d] %s -> %s", i, len(todo), photo_id,
                     label["subject"] if label["gate_category"] == "astronomical_observation"
                     else f"({label['gate_category']})")

        log.info("labelled %d, failed %d", appended, failed)
        return appended

    @task(outlets=[LABEL_LOG])
    def publish_label_log(appended: int) -> int:
        if not appended:
            raise AirflowSkipException("nothing labelled; dataset not rebuilt")
        return appended

    settings = check_backend()
    publish_label_log(label_photos(settings, plan_queue(settings)))


jwst_label()
