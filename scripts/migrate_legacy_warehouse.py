"""
One-off bootstrap: build the warehouse from the jwst-flickr-classifier database.

That project stored embeddings as a column on photos, image paths with the Docker
prefix /usr/local/airflow/include/images/, a keyword-derived canonical_label, and
classifier predictions. This copies photos, their tag labels and embeddings into the
current schema, loads the label log, and checks every image that never received an
embedding. Images that do not decode are marked for download again, which the next
ingest run does. Classifier predictions and training runs are not carried over; the
classifier was trained on the tag labels this project replaced.

Run once, on the host, after moving the legacy images into include/data/images:

    python scripts/migrate_legacy_warehouse.py ../jwst-flickr-classifier/include/jwst.duckdb
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from include.jwst_pipeline import embeddings, images, labelling, quality, warehouse  # noqa: E402
from include.jwst_pipeline.config import paths, utcnow  # noqa: E402
from include.jwst_pipeline.ingest import missing_image_files  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("legacy_db", type=Path)
    args = ap.parse_args()

    with warehouse.session() as con:
        warehouse.migrate(con)
        if con.execute("SELECT count(*) FROM photos").fetchone()[0]:
            sys.exit(f"{paths().warehouse} already holds photos; refusing to migrate over them")

        con.execute(f"ATTACH '{args.legacy_db}' AS legacy (READ_ONLY)")
        con.begin()
        con.execute("""
            INSERT INTO photos (photo_id, title, description, tags, date_taken, first_seen_at,
                                image_file, downloaded_at, download_attempts, legacy_tag_label)
            SELECT photo_id, title, description, tags, date_taken, date_ingested,
                   regexp_extract(image_path, '[^/]+$'), date_ingested, 1, canonical_label
            FROM legacy.photos
        """)
        con.execute(f"""
            INSERT INTO embeddings
            SELECT photo_id, '{embeddings.MODEL_NAME}',
                   CAST(embedding AS FLOAT[{warehouse.EMBEDDING_DIM}]), ?
            FROM legacy.photos WHERE embedding IS NOT NULL
        """, [utcnow()])
        con.commit()
        con.execute("DETACH legacy")

        # Record sizes, and clear image_file where the file is missing or broken so the
        # next ingest downloads it again. A photo with an embedding decoded successfully
        # when it was embedded, so only the rest need a full decode.
        with_embedding = {pid for (pid,) in con.execute("SELECT photo_id FROM embeddings").fetchall()}
        root = paths().images
        broken = []
        for pid, name in con.execute(
                "SELECT photo_id, image_file FROM photos ORDER BY photo_id").fetchall():
            path = root / name
            if not path.exists():
                broken.append((pid, "file missing after migration"))
                continue
            if pid not in with_embedding:
                try:
                    images.verify(path)
                except images.ImageError as exc:
                    broken.append((pid, f"legacy file does not decode: {exc}"))
                    continue
            con.execute("UPDATE photos SET image_bytes = ? WHERE photo_id = ?",
                        [path.stat().st_size, pid])
        for pid, reason in broken:
            con.execute("""UPDATE photos SET image_file = NULL, image_bytes = NULL,
                           downloaded_at = NULL, last_download_error = ? WHERE photo_id = ?""",
                        [reason[:500], pid])

        n_labels = labelling.load_log(con, paths().label_log)

        stored = {n for (n,) in con.execute(
            "SELECT image_file FROM photos WHERE image_file IS NOT NULL").fetchall()}
        orphans = sorted(p.name for p in root.glob("*.jpg") if p.name not in stored
                         and p.name.removesuffix(".jpg") not in {b[0] for b in broken})

        print()
        for label, sql in [
            ("photos", "SELECT count(*) FROM photos"),
            ("  with a usable image", "SELECT count(*) FROM photos WHERE image_file IS NOT NULL"),
            ("  with a tag label", "SELECT count(*) FROM photos WHERE legacy_tag_label IS NOT NULL"),
            ("embeddings", "SELECT count(*) FROM embeddings"),
            ("labels", "SELECT count(*) FROM labels"),
        ]:
            print(f"{label:<24} {con.execute(sql).fetchone()[0]:>6}")
        print(f"{'marked for re-download':<24} {len(broken):>6}")
        for pid, reason in broken:
            print(f"    {pid}  {reason[:80]}")
        print(f"{'files with no photo row':<24} {len(orphans):>6}  {' '.join(orphans)}")
        print(f"{'image files missing':<24} {missing_image_files(con):>6}")
        print()

        quality.run(con, quality.PHOTO_CHECKS + quality.EMBEDDING_CHECKS + quality.LABEL_CHECKS,
                    context="migration")
        print(f"\nwarehouse written to {paths().warehouse} ({n_labels} labels loaded)")


if __name__ == "__main__":
    main()
