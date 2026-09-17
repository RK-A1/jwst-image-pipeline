import numpy as np
import pytest

from conftest import make_image
from include.jwst_pipeline import embeddings
from include.jwst_pipeline.config import paths, utcnow


def test_staged_batch_round_trips(tmp_path):
    vectors = np.random.default_rng(0).random((2, embeddings.DIM)).astype(np.float32)
    path = tmp_path / "batch-00000.npz"
    embeddings.save_batch(path, ["1", "2"], vectors, {"3": "OSError: truncated"})
    ids, loaded, errors = embeddings.load_batch(path)
    assert ids == ["1", "2"] and np.array_equal(loaded, vectors)
    assert errors == {"3": "OSError: truncated"}


def test_store_loads_vectors_and_failures_then_pending_excludes_both(con, add_photo, tmp_path):
    for pid in ["1", "2", "3", "4"]:
        add_photo(pid)
    add_photo("5", with_file=False)
    vectors = np.ones((2, embeddings.DIM), dtype=np.float32)
    batch = tmp_path / "b.npz"
    embeddings.save_batch(batch, ["1", "2"], vectors, {"3": "cannot identify image file"})

    assert embeddings.store(con, [batch], now=utcnow()) == (2, 1)
    assert [r[0] for r in embeddings.pending(con)] == ["4"]


def test_store_is_all_or_nothing(con, add_photo, tmp_path):
    add_photo("1")
    good = tmp_path / "good.npz"
    embeddings.save_batch(good, ["1"], np.ones((1, embeddings.DIM), dtype=np.float32), {})
    missing = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        embeddings.store(con, [good, missing], now=utcnow())
    assert con.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0


def test_embed_produces_one_vector_per_readable_image(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    good = make_image(paths().images / "1.jpg", size=(300, 200))
    bad = paths().images / "2.jpg"
    bad.write_bytes(b"")

    dev = embeddings.device()
    ids, vectors, errors = embeddings.embed([("1", good), ("2", bad)],
                                            embeddings.load_model(dev), dev)
    assert ids == ["1"] and vectors.shape == (1, embeddings.DIM)
    assert set(errors) == {"2"}
