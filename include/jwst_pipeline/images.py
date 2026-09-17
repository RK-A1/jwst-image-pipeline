"""
Image files: atomic downloads, integrity checks and downscaled copies.

The original ingest streamed each download straight to its final path, and skipped any
photo whose file already existed. A download cut off part-way therefore left a
truncated file that was never fetched again. Nine such files, one of them empty, sat
in the archive failing to decode. Here a download goes to a .part file, is checked
against Content-Length, is decoded in full, and only then is renamed into place.
"""

import os
from pathlib import Path

import requests

CHUNK_BYTES = 1 << 20


class ImageError(RuntimeError):
    pass


def open_rgb(path: Path):
    """Open an image as RGB. JWST originals run to 100+ megapixels, past PIL's
    decompression-bomb limit, so that limit is lifted."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as img:
        return img.convert("RGB")


def verify(path: Path) -> None:
    """
    Raise ImageError unless the file decodes to the end.

    Pillow's own verify() only walks the headers, so it passes truncated JPEGs. A full
    load() does not. For JPEGs, draft mode decodes at reduced scale in the DCT domain:
    every byte of the scan is still consumed, at a fraction of the cost.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as img:
            if img.format == "JPEG":
                img.draft("RGB", (256, 256))
            img.load()
    except Exception as exc:
        raise ImageError(f"{path.name} does not decode: {exc}") from exc


def download(url: str, dest: Path, *, session: requests.Session | None = None,
             timeout_s: float = 120.0) -> int:
    """Download url to dest atomically. Returns the size in bytes."""
    http = session or requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with http.get(url, stream=True, timeout=timeout_s) as resp:
            resp.raise_for_status()
            expected = resp.headers.get("Content-Length")
            written = 0
            with part.open("wb") as fh:
                for chunk in resp.iter_content(CHUNK_BYTES):
                    fh.write(chunk)
                    written += len(chunk)
                fh.flush()
                os.fsync(fh.fileno())
        if expected is not None and written != int(expected):
            raise ImageError(f"{dest.name}: received {written} of {expected} bytes")
        verify(part)
        os.replace(part, dest)
        return written
    finally:
        part.unlink(missing_ok=True)


def downscale(src: Path, dest: Path, *, long_edge: int, quality: int) -> None:
    """Write a JPEG copy of src no larger than long_edge on either side."""
    from PIL import Image

    img = open_rgb(src)
    img.thumbnail((long_edge, long_edge), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    img.save(tmp, format="JPEG", quality=quality)
    os.replace(tmp, dest)
