"""
Register manually downloaded photos with the warehouse.

The ingest DAG needs a Flickr API key, but the images themselves can be saved by hand
from a photo's Flickr page. This takes those files, checks each one decodes, copies it
into the images directory under the name the pipeline expects, and records it, so the
embedding and dataset DAGs pick it up as if the ingest had downloaded it.

A photo's caption still comes from the API. Only photos already in the warehouse can be
imported; a file for an unknown photo is reported and skipped, because a label cannot
be produced from an image with no title or caption.

Name each file after the photo id in its Flickr URL — flickr.com/photos/
nasawebbtelescope/51813694550 is 51813694550.jpg. "Large 2048" is ample: the dataset
copies are 1024px and the model is sent 896px.

    python scripts/import_images.py ~/Downloads/jwst-photos
    python scripts/import_images.py ~/Downloads/jwst-photos --dry-run
    python scripts/import_images.py --list-missing
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from include.jwst_pipeline import images, warehouse  # noqa: E402
from include.jwst_pipeline.config import image_file_name, paths, utcnow  # noqa: E402

SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".webp"}


def missing(con) -> list[tuple[str, str]]:
    return con.execute("""
        SELECT photo_id, 'https://www.flickr.com/photos/nasawebbtelescope/' || photo_id
        FROM photos WHERE image_file IS NULL ORDER BY photo_id
    """).fetchall()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("source", nargs="?", type=Path, help="a file, or a directory of files")
    ap.add_argument("--dry-run", action="store_true", help="check the files, change nothing")
    ap.add_argument("--list-missing", action="store_true",
                    help="print the photos with no image, and their Flickr pages")
    args = ap.parse_args()

    with warehouse.session(read_only=args.dry_run or args.list_missing) as con:
        if args.list_missing or not args.source:
            rows = missing(con)
            print(f"{len(rows)} photo(s) have no image file:\n")
            for photo_id, url in rows:
                print(f"  {photo_id}  {url}")
            if not args.source:
                print("\nSave each one as <photo_id>.jpg, then pass the folder to this script.")
            return

        files = ([args.source] if args.source.is_file()
                 else sorted(f for f in args.source.iterdir() if f.suffix.lower() in SUFFIXES))
        if not files:
            sys.exit(f"no image files in {args.source}")

        known = {pid for (pid,) in con.execute("SELECT photo_id FROM photos").fetchall()}
        imported = skipped = failed = 0
        for path in files:
            photo_id = path.stem
            if photo_id not in known:
                print(f"  skip    {path.name}: no photo {photo_id} in the warehouse")
                skipped += 1
                continue
            try:
                images.verify(path)
            except images.ImageError as exc:
                print(f"  FAILED  {path.name}: {exc}")
                failed += 1
                continue

            dest = paths().images / image_file_name(photo_id)
            if args.dry_run:
                print(f"  would import {path.name} -> {dest}")
                imported += 1
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            shutil.copyfile(path, tmp)
            tmp.replace(dest)
            con.execute("""
                UPDATE photos SET image_file = ?, image_bytes = ?, downloaded_at = ?,
                                  last_download_error = NULL
                WHERE photo_id = ?
            """, [dest.name, dest.stat().st_size, utcnow(), photo_id])
            # A new file deserves a fresh embedding attempt.
            con.execute("DELETE FROM embedding_errors WHERE photo_id = ?", [photo_id])
            print(f"  ok      {photo_id}  ({dest.stat().st_size / 1e6:.1f} MB)")
            imported += 1

        print(f"\n{imported} imported, {skipped} skipped, {failed} unreadable")
        if not args.dry_run and imported:
            print(f"{len(missing(con))} photo(s) still have no image.")
            print("Next: trigger jwst_embed, then jwst_dataset.")


if __name__ == "__main__":
    main()
