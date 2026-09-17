import inspect

import pytest

pytest.importorskip("airflow")

from airflow.models.dagbag import DagBag  # noqa: E402

from include.jwst_pipeline import assets  # noqa: E402
from include.jwst_pipeline.config import PROJECT_ROOT, WAREHOUSE_POOL  # noqa: E402


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=str(PROJECT_ROOT / "dags"), include_examples=False)


def test_dags_import_cleanly(dagbag):
    assert dagbag.import_errors == {}
    assert set(dagbag.dag_ids) == {"jwst_ingest", "jwst_embed", "jwst_label", "jwst_dataset"}


def test_every_task_that_opens_the_warehouse_runs_in_its_pool(dagbag):
    """DuckDB accepts one writing process. Outside the one-slot pool, two tasks opening
    the warehouse at once would fail on the file lock."""
    offenders = [
        f"{dag.dag_id}.{t.task_id}"
        for dag in dagbag.dags.values() for t in dag.tasks
        if "warehouse.session" in inspect.getsource(t.python_callable) and t.pool != WAREHOUSE_POOL
    ]
    assert offenders == []


def test_network_tasks_do_not_hold_the_warehouse_pool(dagbag):
    network = {("jwst_ingest", "list_flickr_photos"), ("jwst_ingest", "fetch_metadata"),
               ("jwst_ingest", "download_images"), ("jwst_label", "label_photos"),
               ("jwst_embed", "embed_batch")}
    held = [(d, t) for d, t in network if dagbag.dags[d].get_task(t).pool == WAREHOUSE_POOL]
    assert held == []


def test_dags_are_chained_by_assets(dagbag):
    ingest, embed, label, dataset = (dagbag.dags[d] for d in
                                     ("jwst_ingest", "jwst_embed", "jwst_label", "jwst_dataset"))
    assert ingest.get_task("publish_photos").outlets == [assets.PHOTOS]
    assert embed.schedule == assets.PHOTOS
    assert embed.get_task("publish_embeddings").outlets == [assets.EMBEDDINGS]
    assert label.get_task("publish_label_log").outlets == [assets.LABEL_LOG]
    # Either input rebuilds the dataset. AssetAny has no __eq__, so compare members.
    assert type(dataset.schedule).__name__ == "AssetAny"
    assert {a.name for a in dataset.schedule.objects} == {assets.LABEL_LOG.name, assets.EMBEDDINGS.name}


def test_labelling_spends_money_so_it_never_runs_on_a_schedule(dagbag):
    label = dagbag.dags["jwst_label"]
    assert label.schedule is None
    assert label.params["max_photos"] <= 500


def test_dataset_is_audited_before_it_is_published(dagbag):
    dataset = dagbag.dags["jwst_dataset"]
    assert "audit_dataset" in dataset.get_task("publish_dataset").upstream_task_ids
    assert "load_labels" in dataset.get_task("build_dataset").upstream_task_ids


def test_every_dag_retries_and_allows_one_active_run(dagbag):
    for dag in dagbag.dags.values():
        assert dag.max_active_runs == 1
        assert dag.catchup is False
        assert dag.default_args["retries"] >= 1
