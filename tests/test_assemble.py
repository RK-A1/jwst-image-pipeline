import json

import duckdb
import numpy as np
import pytest

from conftest import label_record
from include.jwst_pipeline import assemble, labelling, quality
from include.jwst_pipeline.config import paths, utcnow


@pytest.mark.parametrize("raw, expected", [
    ("SMACS 0723", "SMACS 0723"),
    ("SMACS 0723.", "SMACS 0723"),
    ("SMACS J0723", "SMACS 0723"),
    ("SMACS 0723.3", "SMACS 0723.3"),     # numerically distinct forms are not merged
    ("Messier 51 (M51)", "M 51"),
    ("m51", "M 51"),
    ("WASP 96b", "WASP-96 b"),
    ("ngc  3132", "NGC 3132"),
    ("Pillars of Creation.", "Pillars of Creation"),
    ("", None),
    (None, None),
])
def test_canonical_name(raw, expected):
    assert assemble.canonical_name(raw) == expected


def test_release_key_groups_renderings_of_one_figure():
    labelled = assemble.release_key("Cosmic Cliffs in Carina (labeled)")
    unlabelled = assemble.release_key("Cosmic Cliffs in Carina (unlabeled inset boxes)")
    square = assemble.release_key("Cosmic Cliffs in Carina [square version]")
    assert labelled == unlabelled == square == "cosmic cliffs in carina"


def test_duplicate_clusters_links_transitively_above_threshold():
    base = np.ones(8, dtype=np.float32)
    near = base.copy()
    near[0] = 1.2
    far = np.arange(8, dtype=np.float32)
    clusters = assemble.duplicate_clusters(["a", "b", "c"], [base, near, far])
    assert clusters["a"] == clusters["b"] != clusters["c"]
    assert assemble.duplicate_clusters([], []) == {}


@pytest.fixture
def small_warehouse(con, add_photo, data_dir):
    """Four labelled photos: a clean and annotated rendering of one release, a second
    release, and a rejected hardware photo."""
    add_photo("10", title="Southern Ring Nebula (NIRCam Image)",
              description="Webb images NGC 3132.", legacy_tag_label="nebula")
    add_photo("11", title="Southern Ring Nebula (labeled)", description="NGC 3132 again.")
    add_photo("20", title="Cassiopeia A", description="A supernova remnant.")
    add_photo("30", title="Mirror polishing", description="Clean room.")

    vec = np.random.default_rng(0).random(2048).astype(np.float32)
    other = np.random.default_rng(1).random(2048).astype(np.float32) * np.linspace(0, 1, 2048)
    for pid, v in [("10", vec), ("11", vec), ("20", other)]:
        con.execute("INSERT INTO embeddings VALUES (?, 'resnet', ?, ?)", [pid, v.tolist(), utcnow()])

    log = paths().label_log
    for record in [
        label_record("10", subject="planetary_nebula", object_name="NGC 3132", instrument="NIRCam"),
        label_record("11", subject="planetary_nebula", modality="annotated_image", object_name="NGC 3132"),
        label_record("20", subject="supernova_remnant", object_name="Cas A"),
        label_record("30", gate_category="hardware_engineering", subject="not_applicable",
                     modality="not_applicable"),
    ]:
        labelling.append_log(log, record)
    labelling.load_log(con, log)
    return con


def read(path, sql="SELECT * FROM t ORDER BY photo_id"):
    c = duckdb.connect()
    c.execute(f"CREATE VIEW t AS SELECT * FROM '{path}'")
    return [dict(zip([d[0] for d in c.description], r)) for r in c.execute(sql).fetchall()]


def test_build_audit_publish(small_warehouse, data_dir):
    staged = paths().staging / "dataset" / "run1"
    assemble.write_images(small_warehouse)
    manifest = assemble.build(small_warehouse, staged)
    assert manifest["rows"] == {"jwst_space_images": 3, "rejected": 1}

    kept = {r["photo_id"]: r for r in read(staged / "jwst_space_images.parquet")}
    assert kept["10"]["release_group"] == kept["11"]["release_group"]
    assert kept["10"]["is_primary"] and not kept["11"]["is_primary"]   # clean image wins
    assert kept["10"]["near_duplicate_group"] == kept["11"]["near_duplicate_group"]
    assert kept["10"]["object_name_verified"] and not kept["20"]["object_name_verified"]
    assert kept["10"]["source_tag_label"] == "nebula"
    assert kept["10"]["image_file"] == "images/10.jpg"
    assert kept["10"]["flickr_url"] == "https://www.flickr.com/photos/nasawebbtelescope/10"
    assert [r["rejected_as"] for r in read(staged / "rejected.parquet")] == ["hardware_engineering"]

    assemble.audit(staged)
    assemble.publish(staged)
    published = paths().dataset
    assert json.loads((published / "manifest.json").read_text()) == manifest
    assert not staged.exists()
    assert sorted(p.name for p in paths().dataset_images.iterdir()) == ["10.jpg", "11.jpg", "20.jpg"]


def test_an_unchanged_build_is_byte_identical(small_warehouse):
    a = assemble.build(small_warehouse, paths().staging / "a")
    b = assemble.build(small_warehouse, paths().staging / "b")
    assert a == b


def test_failed_audit_leaves_the_published_dataset_alone(small_warehouse):
    first = paths().staging / "first"
    assemble.build(small_warehouse, first)
    assemble.audit(first)
    assemble.publish(first)
    before = (paths().dataset / "jwst_space_images.parquet").read_bytes()

    # Relabel a kept photo as a rejection: the build now has fewer rows than published.
    labelling.append_log(paths().label_log, label_record(
        "20", gate_category="artwork_illustration", subject="not_applicable", modality="not_applicable"))
    labelling.load_log(small_warehouse, paths().label_log)
    second = paths().staging / "second"
    assemble.build(small_warehouse, second)

    with pytest.raises(quality.DataQualityError, match="dropped below the published 3"):
        assemble.audit(second)
    assert (paths().dataset / "jwst_space_images.parquet").read_bytes() == before

    assemble.audit(second, allow_shrink=True)


def test_prune_removes_images_of_photos_no_longer_kept(small_warehouse):
    assemble.write_images(small_warehouse)
    labelling.append_log(paths().label_log, label_record(
        "20", gate_category="artwork_illustration", subject="not_applicable", modality="not_applicable"))
    labelling.load_log(small_warehouse, paths().label_log)
    assert assemble.prune_images(small_warehouse) == 1
    assert not (paths().dataset_images / "20.jpg").exists()
