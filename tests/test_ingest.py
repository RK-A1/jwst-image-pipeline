import pytest

from include.jwst_pipeline import images, ingest
from include.jwst_pipeline.flickr import FlickrError


class FakeClient:
    def __init__(self, metadata=None, urls=None):
        self.metadata = metadata or {}
        self.urls = urls or {}

    def photo_metadata(self, photo_id):
        if photo_id not in self.metadata:
            raise FlickrError("Photo not found (code 1)")
        return self.metadata[photo_id]

    def best_image_url(self, photo_id):
        if photo_id not in self.urls:
            raise FlickrError("no sizes")
        return self.urls[photo_id]


def meta(photo_id, title="t", date_taken="2024-05-01 10:00:00"):
    return {"photo_id": photo_id, "title": title, "description": "d",
            "tags": ["jwst"], "date_taken": date_taken}


def test_record_sightings_returns_new_ids_in_listing_order(con, add_photo):
    add_photo("2")
    new = ingest.record_sightings(con, ["3", "2", "1"])
    assert new == ["3", "1"]
    assert con.execute("SELECT last_seen_at IS NOT NULL FROM photos WHERE photo_id = '2'").fetchone()[0]


def test_upsert_updates_captions_but_keeps_first_seen(con):
    ingest.upsert_photos(con, [meta("1", title="old")])
    first_seen = con.execute("SELECT first_seen_at FROM photos").fetchone()[0]
    ingest.upsert_photos(con, [meta("1", title="new")])
    title, again = con.execute("SELECT title, first_seen_at FROM photos").fetchone()
    assert (title, again) == ("new", first_seen)


def test_upsert_accepts_missing_date_taken(con):
    ingest.upsert_photos(con, [meta("1", date_taken=None)])
    assert con.execute("SELECT date_taken FROM photos").fetchone()[0] is None


def test_fetch_metadata_skips_failures_but_not_total_failure():
    client = FakeClient(metadata={"1": meta("1")})
    assert [m["photo_id"] for m in ingest.fetch_metadata(client, ["1", "2"])] == ["1"]
    with pytest.raises(FlickrError):
        ingest.fetch_metadata(client, ["2", "3"])


def test_download_cycle_retries_failures_until_the_attempt_limit(con, monkeypatch, data_dir):
    ingest.upsert_photos(con, [meta("1"), meta("2")])

    def fake_download(url, dest, **kw):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return 1

    monkeypatch.setattr(images, "download", fake_download)
    client = FakeClient(urls={"1": "https://live.staticflickr.com/1.jpg"})

    for attempt in range(ingest.MAX_DOWNLOAD_ATTEMPTS):
        pending = ingest.pending_downloads(con)
        assert "2" in pending
        ingest.record_downloads(con, ingest.download_images(client, pending))

    assert ingest.pending_downloads(con) == []
    rows = dict(con.execute("SELECT photo_id, image_file FROM photos").fetchall())
    assert rows == {"1": "1.jpg", "2": None}
    attempts, error = con.execute(
        "SELECT download_attempts, last_download_error FROM photos WHERE photo_id = '2'").fetchone()
    assert attempts == ingest.MAX_DOWNLOAD_ATTEMPTS and "no sizes" in error


def test_a_fresh_download_clears_an_embedding_failure(con, monkeypatch):
    from include.jwst_pipeline.config import utcnow

    ingest.upsert_photos(con, [meta("1")])
    con.execute("INSERT INTO embedding_errors VALUES ('1', 'truncated', ?)", [utcnow()])
    ingest.record_downloads(con, [{"photo_id": "1", "image_file": "1.jpg", "image_bytes": 10}])
    assert con.execute("SELECT count(*) FROM embedding_errors").fetchone()[0] == 0


def test_unexpected_errors_are_recorded_without_their_message(monkeypatch):
    def boom(url, dest, **kw):
        raise RuntimeError("https://api.flickr.com/?api_key=SECRET")

    monkeypatch.setattr(images, "download", boom)
    [result] = ingest.download_images(FakeClient(urls={"1": "u"}), ["1"])
    assert result == {"photo_id": "1", "error": "RuntimeError"}


def test_missing_image_files_are_counted(con, add_photo, data_dir):
    add_photo("1")
    add_photo("2")
    (data_dir / "images" / "2.jpg").unlink()
    assert ingest.missing_image_files(con) == 1
