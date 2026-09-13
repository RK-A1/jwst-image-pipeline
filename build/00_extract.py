"""
00_extract.py — pull candidate records out of the JWST pipeline database.

Read-only. Nothing is written back to the source project.

Candidate set is the UNION of two signals, because neither is trustworthy alone:

  A. canonical_label IS NOT NULL   — the tag-based labeller reached it
  B. date_taken >= 2021-12-25      — on or after launch

Date alone drops real observations whose Flickr `date_taken` is wrong (the Horsehead
Nebula and Jupiter's Great Red Spot both carry 2124 dates). The tag labels alone miss
recent photos the labeller never ran on. The union is ~1,077 rows; the model gates it
down from there.

Outputs:
  data/candidates.parquet   one row per candidate, no embeddings
  data/embeddings.parquet   photo_id + 2048-dim ResNet50 vector, kept separate so
                            candidates.parquet stays small enough to eyeball
"""

import sys
from pathlib import Path

import duckdb

SOURCE_DB = Path("/Users/rk/ds/JWST/include/jwst.duckdb")
FLICKR_USER = "nasawebbtelescope"   # the account the source project ingests
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

# Union predicate. Keep in sync with the README.
CANDIDATE_WHERE = """
    canonical_label IS NOT NULL
    OR (date_taken >= '2021-12-25' AND date_taken < '2100-01-01')
"""


def main() -> None:
    if not SOURCE_DB.exists():
        sys.exit(f"Source database not found: {SOURCE_DB}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        con = duckdb.connect(str(SOURCE_DB), read_only=True)
    except duckdb.IOException as exc:
        sys.exit(
            f"Could not open {SOURCE_DB} read-only: {exc}\n"
            "A writer probably holds the lock — stop Airflow (`astro dev stop`) and retry."
        )

    total = con.execute("SELECT count(*) FROM photos").fetchone()[0]
    n = con.execute(f"SELECT count(*) FROM photos WHERE {CANDIDATE_WHERE}").fetchone()[0]
    print(f"source photos: {total}")
    print(f"candidates:    {n}")

    # image_path in the source DB carries the Docker-internal prefix; rewrite it to
    # the host path so downstream steps can actually open the file.
    con.execute(f"""
        COPY (
            SELECT
                photo_id,
                title,
                description,
                tags                                        AS flickr_tags,
                date_taken,
                canonical_label                             AS source_tag_label,
                predicted_label                             AS source_predicted_label,
                -- The source DB stores no URL, but Flickr photo pages are
                -- deterministic from the ID, so this costs nothing to reconstruct.
                'https://www.flickr.com/photos/' || '{FLICKR_USER}' || '/' || photo_id
                                                            AS flickr_url,
                '/Users/rk/ds/JWST/include/images/' ||
                    regexp_extract(image_path, '[^/]+$')    AS source_image_path,
                embedding IS NOT NULL                       AS has_embedding
            FROM photos
            WHERE {CANDIDATE_WHERE}
            ORDER BY photo_id
        ) TO '{OUT_DIR / "candidates.parquet"}' (FORMAT parquet)
    """)

    con.execute(f"""
        COPY (
            SELECT photo_id, embedding
            FROM photos
            WHERE ({CANDIDATE_WHERE}) AND embedding IS NOT NULL
            ORDER BY photo_id
        ) TO '{OUT_DIR / "embeddings.parquet"}' (FORMAT parquet)
    """)

    # Report on what came out, and flag anything the later steps will trip over.
    missing_desc = con.execute(
        f"SELECT count(*) FROM photos WHERE ({CANDIDATE_WHERE}) "
        "AND (description IS NULL OR trim(description) = '')"
    ).fetchone()[0]
    missing_file = sum(
        0 if Path(p).exists() else 1
        for (p,) in con.execute(
            "SELECT '/Users/rk/ds/JWST/include/images/' || regexp_extract(image_path, '[^/]+$') "
            f"FROM photos WHERE {CANDIDATE_WHERE}"
        ).fetchall()
    )
    con.close()

    for f in ("candidates.parquet", "embeddings.parquet"):
        print(f"  wrote data/{f}  ({(OUT_DIR / f).stat().st_size / 1e6:.1f} MB)")
    print(f"  candidates with no description: {missing_desc}")
    print(f"  candidates with no image file:  {missing_file}")


if __name__ == "__main__":
    main()
