import pytest
import requests

from include.jwst_pipeline.flickr import FlickrClient, FlickrError, parse_photo_info, pick_size

KEY = "0123456789abcdef0123456789abcdef"


class FakeResponse:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self._text = text

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """Plays back a scripted sequence of responses or exceptions."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(*script) -> tuple[FlickrClient, FakeSession]:
    session = FakeSession(*script)
    return FlickrClient(KEY, session=session, sleep=lambda s: None), session


def ok(**body):
    return FakeResponse(body={"stat": "ok", **body})


def test_list_photo_ids_paginates_and_deduplicates():
    c, session = client(
        ok(photos={"page": 1, "pages": 2, "total": "3", "photo": [{"id": "3"}, {"id": "2"}]}),
        ok(photos={"page": 2, "pages": 2, "total": "3", "photo": [{"id": "2"}, {"id": "1"}]}),
    )
    assert c.list_photo_ids("nsid") == ["3", "2", "1"]
    assert [p["page"] for p in session.calls] == [1, 2]


def test_transient_errors_are_retried():
    c, session = client(FakeResponse(status=503), requests.Timeout(), ok(user={"id": "nsid"}))
    assert c.resolve_user_id() == "nsid"
    assert len(session.calls) == 3


def test_permanent_api_errors_are_not_retried():
    c, session = client(FakeResponse(body={"stat": "fail", "code": 1, "message": "Photo not found"}))
    with pytest.raises(FlickrError, match="Photo not found") as exc:
        c.photo_metadata("404")
    assert exc.value.transient is False
    assert len(session.calls) == 1


def test_gives_up_after_max_attempts():
    c, session = client(*[FakeResponse(status=500)] * 4)
    with pytest.raises(FlickrError, match="HTTP 500"):
        c.resolve_user_id()
    assert len(session.calls) == 4


def test_api_key_never_appears_in_errors():
    url = f"https://api.flickr.com/services/rest/?api_key={KEY}&method=x"
    c, _ = client(*[requests.ConnectionError(f"Max retries exceeded with url: {url}")] * 4)
    with pytest.raises(FlickrError) as exc:
        c.resolve_user_id()
    assert KEY not in str(exc.value)
    # Airflow logs chained exceptions, and the original message holds the key.
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


def test_parse_photo_info():
    info = {
        "id": "55530498785",
        "title": {"_content": "Webb panorama"},
        "description": {"_content": "<b>caption</b>"},
        "tags": {"tag": [{"_content": "starcluster"}, {"_content": "jwst"}]},
        "dates": {"taken": "2026-09-15 12:00:00"},
    }
    assert parse_photo_info(info) == {
        "photo_id": "55530498785", "title": "Webb panorama", "description": "<b>caption</b>",
        "tags": ["starcluster", "jwst"], "date_taken": "2026-09-15 12:00:00",
    }


def test_pick_size_prefers_original_then_falls_back():
    sizes = [{"label": "Medium", "source": "m"}, {"label": "Large 2048", "source": "l"},
             {"label": "Original", "source": "o"}]
    assert pick_size(sizes) == "o"
    assert pick_size(sizes[:2]) == "l"
    assert pick_size([{"label": "Square", "source": "sq"}]) == "sq"
    with pytest.raises(FlickrError):
        pick_size([])
