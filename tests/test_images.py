import io

import pytest
from PIL import Image

from conftest import make_image
from include.jwst_pipeline import images


def jpeg_bytes(size=(320, 240)) -> bytes:
    buf = io.BytesIO()
    # Noise rather than a flat colour, so the entropy-coded scan is long enough to cut.
    Image.effect_noise(size, 60).convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class StreamResponse:
    def __init__(self, body: bytes, content_length: int | None):
        self.body = body
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk):
        for i in range(0, len(self.body), chunk):
            yield self.body[i:i + chunk]


class StreamSession:
    def __init__(self, response):
        self.response = response

    def get(self, url, stream=False, timeout=None):
        return self.response


def test_verify_accepts_complete_images(tmp_path):
    images.verify(make_image(tmp_path / "a.jpg"))
    images.verify(make_image(tmp_path / "b.png", fmt="PNG"))


def test_verify_rejects_truncated_and_empty_files(tmp_path):
    truncated = tmp_path / "t.jpg"
    data = jpeg_bytes()
    truncated.write_bytes(data[: len(data) // 2])
    with pytest.raises(images.ImageError):
        images.verify(truncated)

    empty = tmp_path / "e.jpg"
    empty.write_bytes(b"")
    with pytest.raises(images.ImageError):
        images.verify(empty)


def test_download_writes_atomically(tmp_path):
    body = jpeg_bytes()
    dest = tmp_path / "images" / "1.jpg"
    size = images.download("u", dest, session=StreamSession(StreamResponse(body, len(body))))
    assert size == len(body) and dest.read_bytes() == body
    assert not list(dest.parent.glob("*.part"))


def test_download_rejects_short_body_and_leaves_nothing(tmp_path):
    body = jpeg_bytes()
    dest = tmp_path / "1.jpg"
    with pytest.raises(images.ImageError, match="received"):
        images.download("u", dest, session=StreamSession(StreamResponse(body[:100], len(body))))
    assert not dest.exists() and not list(tmp_path.glob("*.part"))


def test_download_without_content_length_is_still_decoded(tmp_path):
    body = jpeg_bytes()
    dest = tmp_path / "1.jpg"
    with pytest.raises(images.ImageError, match="does not decode"):
        images.download("u", dest, session=StreamSession(StreamResponse(body[: len(body) // 2], None)))
    assert not dest.exists()


def test_download_never_replaces_a_good_file_with_a_bad_one(tmp_path):
    dest = make_image(tmp_path / "1.jpg")
    before = dest.read_bytes()
    with pytest.raises(images.ImageError):
        images.download("u", dest, session=StreamSession(StreamResponse(b"junk", None)))
    assert dest.read_bytes() == before


def test_downscale_bounds_the_long_edge(tmp_path):
    src = make_image(tmp_path / "big.png", size=(3000, 1200), fmt="PNG")
    dest = tmp_path / "out" / "small.jpg"
    images.downscale(src, dest, long_edge=1024, quality=88)
    with Image.open(dest) as img:
        assert img.format == "JPEG" and max(img.size) == 1024
