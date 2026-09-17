"""
Ingest steps, independent of Airflow.

Each step either talks to Flickr or writes the warehouse, never both, so the DAG can
run network work outside the warehouse pool. What to download is decided from
warehouse state rather than passed along from the previous step, so a failed
download is retried on a later run without anyone intervening.
"""

import logging

from include.jwst_pipeline import images
from include.jwst_pipeline.config import image_file_name, paths, utcnow
from include.jwst_pipeline.flickr import FlickrClient, FlickrError

log = logging.getLogger(__name__)

MAX_DOWNLOAD_ATTEMPTS = 3


def record_sightings(con, flickr_ids: list[str]) -> list[str]:
    """Stamp last_seen_at on every listed photo and return the ids not yet stored."""
    now = utcnow()
    con.execute("CREATE OR REPLACE TEMP TABLE seen (photo_id VARCHAR)")
    if flickr_ids:
        con.executemany("INSERT INTO seen VALUES (?)", [(i,) for i in flickr_ids])
    con.execute("UPDATE photos SET last_seen_at = ? WHERE photo_id IN (SELECT photo_id FROM seen)",
                [now])
    known = {pid for (pid,) in con.execute("SELECT photo_id FROM photos").fetchall()}
    new = [pid for pid in flickr_ids if pid not in known]
    log.info("Flickr lists %d photos; %d already stored, %d new",
             len(flickr_ids), len(flickr_ids) - len(new), len(new))
    return new


def fetch_metadata(client: FlickrClient, photo_ids: list[str]) -> list[dict]:
    """
    Caption metadata for each id. A photo that fails is left out, and since it is then
    still absent from the warehouse, the next run lists it as new and tries again.
    """
    out, failures = [], 0
    for photo_id in photo_ids:
        try:
            out.append(client.photo_metadata(photo_id))
        except FlickrError as exc:
            failures += 1
            log.warning("metadata for %s failed: %s", photo_id, exc)
    if photo_ids and not out:
        raise FlickrError(f"metadata failed for all {len(photo_ids)} new photos")
    log.info("fetched metadata for %d of %d photos (%d failed)", len(out), len(photo_ids), failures)
    return out


def upsert_photos(con, metadata: list[dict]) -> int:
    now = utcnow()
    for m in metadata:
        con.execute("""
            INSERT INTO photos (photo_id, title, description, tags, date_taken,
                                first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, CAST(? AS TIMESTAMP), ?, ?)
            ON CONFLICT (photo_id) DO UPDATE SET
                title = excluded.title, description = excluded.description,
                tags = excluded.tags, date_taken = excluded.date_taken,
                last_seen_at = excluded.last_seen_at
        """, [m["photo_id"], m["title"], m["description"], m["tags"], m["date_taken"], now, now])
    return len(metadata)


def pending_downloads(con) -> list[str]:
    return [pid for (pid,) in con.execute("""
        SELECT photo_id FROM photos
        WHERE image_file IS NULL AND download_attempts < ?
        ORDER BY photo_id
    """, [MAX_DOWNLOAD_ATTEMPTS]).fetchall()]


def download_images(client: FlickrClient, photo_ids: list[str]) -> list[dict]:
    """Download each photo's largest rendition. One result per id, success or not."""
    results = []
    for photo_id in photo_ids:
        name = image_file_name(photo_id)
        try:
            url = client.best_image_url(photo_id)
            size = images.download(url, paths().images / name)
            results.append({"photo_id": photo_id, "image_file": name, "image_bytes": size})
        except Exception as exc:
            # Never str() a requests exception into the result: it can hold the API key.
            message = str(exc) if isinstance(exc, (FlickrError, images.ImageError)) else type(exc).__name__
            log.warning("download of %s failed: %s", photo_id, message)
            results.append({"photo_id": photo_id, "error": message})
    return results


def record_downloads(con, results: list[dict]) -> tuple[int, int]:
    now = utcnow()
    ok = failed = 0
    for r in results:
        if "error" in r:
            con.execute("""
                UPDATE photos SET download_attempts = download_attempts + 1,
                                  last_download_error = ?
                WHERE photo_id = ?
            """, [r["error"][:500], r["photo_id"]])
            failed += 1
        else:
            con.execute("""
                UPDATE photos SET image_file = ?, image_bytes = ?, downloaded_at = ?,
                                  download_attempts = download_attempts + 1,
                                  last_download_error = NULL
                WHERE photo_id = ?
            """, [r["image_file"], r["image_bytes"], now, r["photo_id"]])
            # A fresh file deserves a fresh embedding attempt.
            con.execute("DELETE FROM embedding_errors WHERE photo_id = ?", [r["photo_id"]])
            ok += 1
    return ok, failed


def missing_image_files(con) -> int:
    """Photos marked downloaded whose file is not on disk."""
    root = paths().images
    return sum(
        1 for (name,) in con.execute(
            "SELECT image_file FROM photos WHERE image_file IS NOT NULL").fetchall()
        if not (root / name).exists()
    )
