"""
Build the published dataset from the warehouse, using write-audit-publish.

    build()    writes every output file into a staging directory
    audit()    runs quality checks against the staged files
    publish()  swaps the staged files into the dataset directory, all at once

Consumers of the dataset directory therefore never see a half-written or unaudited
build. A failed audit leaves the published dataset exactly as it was.
"""

import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path

import duckdb

from include.jwst_pipeline import codebook, images, quality
from include.jwst_pipeline.config import FLICKR_USER, image_file_name, paths

log = logging.getLogger(__name__)

IMAGE_LONG_EDGE = 1024
IMAGE_QUALITY = 88

KEPT = "jwst_space_images"
REJECTED = "rejected"
OUTPUT_FILES = [f"{KEPT}.parquet", f"{KEPT}.csv", f"{REJECTED}.parquet", f"{REJECTED}.csv",
                "manifest.json"]


# ── object names ────────────────────────────────────────────────────────────────

def normalise(s: str | None) -> str:
    """Lowercase, strip punctuation and whitespace — for verbatim name matching."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# Catalogue prefixes seen in these captions, longest first so "TRAPPIST" wins over "T".
_CATALOGUES = [
    "TRAPPIST", "Messier", "SMACS", "Abell", "WASP", "NGC", "Arp", "HH", "IC", "M",
]
_CAT_RE = re.compile(
    r"^(" + "|".join(_CATALOGUES) + r")\s*[-–]?\s*(J)?\s*([0-9][0-9.\-]*)\s*([a-z])?\s*$",
    re.IGNORECASE,
)
_CANONICAL_CASE = {c.upper(): c for c in _CATALOGUES}
# "Messier 51" and "M51" name the same catalogue entry — fold the long form onto the
# short one so they do not split into two rows under GROUP BY object_name_normalized.
_CANONICAL_CASE["MESSIER"] = "M"
# A trailing parenthetical gloss — "Messier 51 (M51)" — otherwise defeats the match.
_PAREN_TAIL = re.compile(r"\s*\([^)]*\)\s*$")


def canonical_name(s: str | None) -> str | None:
    """
    Canonical display form of a catalogue designation, so the same object does not
    appear under several spellings. These captions spell one cluster four ways —
    "SMACS 0723", "SMACS 0723.", "SMACS 0723.3", "SMACS J0723".

    Normalises: surrounding and internal whitespace, trailing punctuation, the
    optional "J" coordinate prefix, prefix capitalisation, and the spacing between
    prefix and number. Deliberately does NOT unify numerically distinct forms
    (0723 vs 0723.3) — truncating digits can merge genuinely different objects.

    Anything that is not a recognised catalogue designation (proper names like
    "Pillars of Creation") is returned with whitespace collapsed and trailing
    punctuation stripped, otherwise unchanged.
    """
    if not s:
        return None
    cleaned = re.sub(r"\s+", " ", s).strip().rstrip(".,;:")
    if not cleaned:
        return None
    stripped = _PAREN_TAIL.sub("", cleaned)
    if stripped:
        cleaned = stripped

    m = _CAT_RE.match(cleaned)
    if not m:
        return cleaned

    prefix, _j, number, letter = m.groups()
    prefix = _CANONICAL_CASE.get(prefix.upper(), prefix)
    number = number.rstrip(".")
    # Exoplanet host catalogues hyphenate ("WASP-96 b"); the deep-sky ones use a
    # space ("NGC 3132"). Keeping the conventional form matters for joining to
    # SIMBAD or NED later.
    sep = "-" if prefix in {"WASP", "TRAPPIST"} else " "
    out = f"{prefix}{sep}{number}"
    if letter:
        out += f" {letter.lower()}"
    return out


# ── instrument ──────────────────────────────────────────────────────────────────

# Each Webb instrument by acronym or full name, as captions use both.
_INSTRUMENT_PATTERNS = {
    "NIRCam": r"\bNIRCam\b|\bnear[\s-]?infrared\s+camera\b",
    "MIRI": r"\bMIRI\b|\bmid[\s-]?infrared\s+instrument\b",
    "NIRSpec": r"\bNIRSpec\b|\bnear[\s-]?infrared\s+spectrograph\b",
    "NIRISS": r"\bNIRISS\b|\bnear[\s-]?infrared\s+imager\s+and\s+slitless\s+spectrograph\b",
    "FGS": r"\bFGS\b|\bfine\s+guidance\s+sensor\b",
}


def named_instruments(title: str | None, description: str | None) -> list[str]:
    from include.jwst_pipeline.labelling import clean_caption

    text = f"{title or ''} {clean_caption(description)}"
    return [name for name, pattern in _INSTRUMENT_PATTERNS.items()
            if re.search(pattern, text, re.IGNORECASE)]


def correct_instrument(value: str, title: str | None, description: str | None) -> str:
    """
    Enforce the codebook's rule that `multiple` needs two or more Webb instruments named
    in the caption. The model reaches for `multiple` on multi-observatory releases
    ("Hubble and Webb") that name none, so a `multiple` backed by fewer than two names
    becomes the one instrument named, or `unknown`. Other values are left alone: this
    only ever removes a claim the text does not support.
    """
    if value != "multiple":
        return value
    named = named_instruments(title, description)
    if len(named) >= 2:
        return "multiple"
    return named[0] if named else "unknown"


# ── grouping ────────────────────────────────────────────────────────────────────

_QUALIFIER = re.compile(r"\s*[\(\[][^)\]]*[)\]]\s*")


def release_key(title: str | None) -> str:
    """
    Collapse a title to its press-release identity by dropping the parenthesised
    qualifier that distinguishes renderings of the same figure — "(NIRCam Image)",
    "(labeled)", "(unlabeled inset boxes)", "(Hubble + Webb)", "[square version]".

    This is the grouping that matters most. Photo 52619016131 "(unlabeled inset
    boxes)" and 52619510618 "(labeled)" are the same figure with overlays added, but
    their ResNet embeddings are only 0.83 apart — annotation changes enough visual
    texture that pixel similarity misses them. The title does not.
    """
    t = _QUALIFIER.sub(" ", title or "")
    t = re.sub(r"[^a-z0-9 ]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def duplicate_clusters(ids: list[str], vecs: list, threshold: float = 0.95) -> dict[str, int]:
    """
    Single-linkage clusters over cosine similarity of the ResNet embeddings.

    Complements release_key rather than replacing it: this catches the same picture
    republished under a *different* headline, where the title gives nothing away.
    Kept tight at 0.95 — by 0.85 the bands start merging genuinely different
    observations that merely look alike.
    """
    import numpy as np

    if not ids:
        return {}
    M = np.asarray(vecs, dtype=np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    S = M @ M.T
    np.fill_diagonal(S, 0.0)

    cluster = [-1] * len(ids)
    cid = 0
    for i in range(len(ids)):
        if cluster[i] >= 0:
            continue
        stack, cluster[i] = [i], cid
        while stack:
            x = stack.pop()
            for y in np.where(S[x] >= threshold)[0]:
                if cluster[y] < 0:
                    cluster[y] = cid
                    stack.append(y)
        cid += 1
    return dict(zip(ids, cluster))


# Prefer a clean image as the primary of a release group, then the largest original.
_MODALITY_RANK = {"image": 0, "annotated_image": 1, "comparison_composite": 2, "spectrum_or_plot": 3}


# ── build ───────────────────────────────────────────────────────────────────────

def build(con: duckdb.DuckDBPyConnection, out_dir: Path) -> dict:
    """Write every dataset file into out_dir. Returns the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = f"l.gate_category = '{codebook.KEEP_GATE}'"

    # Object names: verified if they appear verbatim in the caption, then canonicalised.
    named = con.execute("""
        SELECT p.photo_id, p.title, p.description, l.object_name
        FROM photos p JOIN labels l USING (photo_id)
        WHERE l.object_name IS NOT NULL
    """).fetchall()
    con.execute("CREATE OR REPLACE TEMP TABLE verif "
                "(photo_id VARCHAR, object_name_verified BOOLEAN, object_name_normalized VARCHAR)")
    if named:
        con.executemany("INSERT INTO verif VALUES (?, ?, ?)", [
            (pid, normalise(name) in normalise(f"{title} {desc}"), canonical_name(name))
            for pid, title, desc, name in named
        ])

    # The model's raw instrument stays in the label log; the dataset gets the corrected one.
    multiple = con.execute("""
        SELECT p.photo_id, p.title, p.description
        FROM photos p JOIN labels l USING (photo_id)
        WHERE l.instrument = 'multiple'
    """).fetchall()
    con.execute("CREATE OR REPLACE TEMP TABLE instrument_fix (photo_id VARCHAR, instrument VARCHAR)")
    if multiple:
        con.executemany("INSERT INTO instrument_fix VALUES (?, ?)", [
            (pid, correct_instrument("multiple", title, desc)) for pid, title, desc in multiple
        ])

    # Three levels of grouping, because each misses what the others catch:
    #   object_name_normalized : every image of this object          (coarsest)
    #   release_group          : one press release / figure set      (the useful one)
    #   near_duplicate_group   : literally the same pixels           (rare, 2%)
    kept_rows = con.execute(f"""
        SELECT p.photo_id, p.title, l.modality, coalesce(p.image_bytes, 0)
        FROM photos p JOIN labels l USING (photo_id)
        WHERE {keep} ORDER BY p.photo_id
    """).fetchall()
    release = {pid: release_key(title) for pid, title, _, _ in kept_rows}

    emb = con.execute(f"""
        SELECT e.photo_id, e.vector FROM embeddings e JOIN labels l USING (photo_id)
        WHERE {keep} ORDER BY e.photo_id
    """).fetchall()
    dup = duplicate_clusters([r[0] for r in emb], [r[1] for r in emb])

    by_release: dict[str, list] = {}
    for pid, _, modality, size in kept_rows:
        by_release.setdefault(release[pid], []).append((_MODALITY_RANK.get(modality, 9), -size, pid))
    primary = {sorted(v)[0][2] for v in by_release.values()}

    con.execute("CREATE OR REPLACE TEMP TABLE grouping (photo_id VARCHAR, release_group VARCHAR, "
                "near_duplicate_group INTEGER, is_primary BOOLEAN)")
    if kept_rows:
        con.executemany("INSERT INTO grouping VALUES (?, ?, ?, ?)", [
            (pid, release[pid], dup.get(pid), pid in primary) for pid, _, _, _ in kept_rows
        ])

    dataset_images = paths().dataset_images
    present = {p.name for p in dataset_images.glob("*.jpg")} if dataset_images.exists() else set()
    con.execute("CREATE OR REPLACE TEMP TABLE images_present (image_file VARCHAR)")
    if present:
        con.executemany("INSERT INTO images_present VALUES (?)", [(n,) for n in sorted(present)])

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW labelled AS
        SELECT
            p.photo_id, p.title, p.description,
            l.subject, l.modality, l.object_name,
            v.object_name_normalized,
            coalesce(v.object_name_verified, l.object_name IS NULL) AS object_name_verified,
            coalesce(f.instrument, l.instrument)              AS instrument,
            l.confidence, l.rationale, l.gate_category,
            g.release_group, g.near_duplicate_group, g.is_primary,
            p.tags                                            AS flickr_tags,
            p.date_taken,
            'https://www.flickr.com/photos/{FLICKR_USER}/' || p.photo_id AS flickr_url,
            p.legacy_tag_label                                AS source_tag_label,
            CASE WHEN i.image_file IS NOT NULL
                 THEN 'images/' || i.image_file END           AS image_file,
            e.photo_id IS NOT NULL                            AS has_embedding,
            l.model, l.codebook_version, l.vision             AS labelled_with_vision
        FROM photos p
        JOIN labels l USING (photo_id)
        LEFT JOIN verif v USING (photo_id)
        LEFT JOIN instrument_fix f USING (photo_id)
        LEFT JOIN grouping g USING (photo_id)
        LEFT JOIN embeddings e USING (photo_id)
        LEFT JOIN images_present i ON i.image_file = p.photo_id || '.jpg'
    """)

    kept_cols = """photo_id, title, description, subject, modality, object_name,
        object_name_normalized, object_name_verified, instrument, confidence, rationale,
        release_group, near_duplicate_group, is_primary, flickr_tags,
        date_taken, flickr_url, source_tag_label, image_file, has_embedding,
        model, codebook_version, labelled_with_vision"""
    con.execute(f"""
        COPY (SELECT {kept_cols} FROM labelled WHERE gate_category = '{codebook.KEEP_GATE}'
              ORDER BY photo_id)
        TO '{out_dir / f"{KEPT}.parquet"}' (FORMAT parquet)
    """)
    con.execute(f"""
        COPY (SELECT photo_id, title, gate_category AS rejected_as, rationale,
                     date_taken, flickr_url, source_tag_label
              FROM labelled WHERE gate_category <> '{codebook.KEEP_GATE}' ORDER BY photo_id)
        TO '{out_dir / f"{REJECTED}.parquet"}' (FORMAT parquet)
    """)

    # CSVs for eyeballing in a spreadsheet: useful columns first, one line per row.
    csv_cols = """
        photo_id, subject, modality, object_name_normalized, object_name,
        object_name_verified, instrument, title, flickr_url, date_taken,
        source_tag_label, rationale, confidence,
        release_group, near_duplicate_group, is_primary,
        array_to_string(flickr_tags, '|')                       AS flickr_tags,
        regexp_replace(description, '\\s*\\n+\\s*', ' ', 'g')     AS description,
        has_embedding, model, codebook_version
    """
    con.execute(f"""
        COPY (SELECT {csv_cols} FROM labelled WHERE gate_category = '{codebook.KEEP_GATE}'
              ORDER BY subject, photo_id)
        TO '{out_dir / f"{KEPT}.csv"}' (HEADER, DELIMITER ',')
    """)
    con.execute(f"""
        COPY (SELECT photo_id, gate_category AS rejected_as, title, rationale,
                     flickr_url, date_taken, source_tag_label
              FROM labelled WHERE gate_category <> '{codebook.KEEP_GATE}'
              ORDER BY gate_category, photo_id)
        TO '{out_dir / f"{REJECTED}.csv"}' (HEADER, DELIMITER ',')
    """)

    manifest = _manifest(con, out_dir)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _log_summary(con)
    return manifest


def _manifest(con: duckdb.DuckDBPyConnection, out_dir: Path) -> dict:
    """Counts and checksums only, no timestamps, so an unchanged build is byte-identical."""
    def counts(sql: str) -> dict:
        return {str(k): v for k, v in con.execute(sql).fetchall()}

    return {
        "rows": {
            KEPT: con.execute(f"SELECT count(*) FROM '{out_dir / f'{KEPT}.parquet'}'").fetchone()[0],
            REJECTED: con.execute(f"SELECT count(*) FROM '{out_dir / f'{REJECTED}.parquet'}'").fetchone()[0],
        },
        "labels": con.execute("SELECT count(*) FROM labels").fetchone()[0],
        "subjects": counts(f"SELECT subject, count(*) FROM labelled "
                           f"WHERE gate_category = '{codebook.KEEP_GATE}' GROUP BY 1 ORDER BY 1"),
        "codebook_versions": counts("SELECT codebook_version, count(*) FROM labels GROUP BY 1 ORDER BY 1"),
        "embedding_model": "resnet50-imagenet1k-v2",
        "sha256": {
            name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
            for name in OUTPUT_FILES if name != "manifest.json"
        },
    }


def _log_summary(con: duckdb.DuckDBPyConnection) -> None:
    for title, sql in [
        ("rejected by category", f"SELECT gate_category, count(*) FROM labelled "
                                 f"WHERE gate_category <> '{codebook.KEEP_GATE}' GROUP BY 1 ORDER BY 2 DESC"),
        ("subject", f"SELECT subject, count(*) FROM labelled "
                    f"WHERE gate_category = '{codebook.KEEP_GATE}' GROUP BY 1 ORDER BY 2 DESC"),
        ("modality", f"SELECT modality, count(*) FROM labelled "
                     f"WHERE gate_category = '{codebook.KEEP_GATE}' GROUP BY 1 ORDER BY 2 DESC"),
        ("instrument", f"SELECT instrument, count(*) FROM labelled "
                       f"WHERE gate_category = '{codebook.KEEP_GATE}' GROUP BY 1 ORDER BY 2 DESC"),
    ]:
        rows = con.execute(sql).fetchall()
        log.info("%s: %s", title, ", ".join(f"{k} {n}" for k, n in rows))


# ── images ──────────────────────────────────────────────────────────────────────

def write_images(con: duckdb.DuckDBPyConnection) -> tuple[int, int, int]:
    """
    Downscaled copies of every kept photo that does not have one yet.
    Returns (written, already present, failed).
    """
    rows = con.execute(f"""
        SELECT p.photo_id, p.image_file FROM photos p JOIN labels l USING (photo_id)
        WHERE l.gate_category = '{codebook.KEEP_GATE}' ORDER BY p.photo_id
    """).fetchall()
    originals, out = paths().images, paths().dataset_images
    written = present = failed = 0
    for photo_id, image_file in rows:
        dest = out / image_file_name(photo_id)
        if dest.exists():
            present += 1
            continue
        if image_file is None:
            failed += 1
            continue
        try:
            images.downscale(originals / image_file, dest,
                             long_edge=IMAGE_LONG_EDGE, quality=IMAGE_QUALITY)
            written += 1
        except Exception as exc:
            log.warning("downscale failed for %s: %s", photo_id, exc)
            failed += 1
    return written, present, failed


# ── audit and publish ───────────────────────────────────────────────────────────

def audit(staged: Path, *, allow_shrink: bool = False) -> list[quality.Result]:
    manifest = json.loads((staged / "manifest.json").read_text())
    published = paths().dataset / "manifest.json"
    published_rows = json.loads(published.read_text())["rows"][KEPT] if published.exists() else None

    con = duckdb.connect()
    try:
        con.execute(f"CREATE VIEW kept AS SELECT * FROM '{staged / f'{KEPT}.parquet'}'")
        con.execute(f"CREATE VIEW rejected AS SELECT * FROM '{staged / f'{REJECTED}.parquet'}'")
        checks = quality.dataset_checks(manifest["labels"], published_rows, allow_shrink)
        checks.append(quality.Check(
            "CSV and parquet row counts agree",
            lambda c: abs(
                c.execute(f"SELECT count(*) FROM read_csv('{staged / f'{KEPT}.csv'}')").fetchone()[0]
                - manifest["rows"][KEPT]
            ),
        ))
        return quality.run(con, checks, context="dataset")
    finally:
        con.close()


def publish(staged: Path) -> None:
    """
    Move staged files into the dataset directory. Each os.replace is atomic, and the
    manifest goes last, so it only ever describes files that are already in place.
    """
    dest = paths().dataset
    dest.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        os.replace(staged / name, dest / name)
    shutil.rmtree(staged, ignore_errors=True)


def prune_images(con: duckdb.DuckDBPyConnection) -> int:
    """Remove downscaled copies of photos that are no longer in the dataset."""
    kept = {image_file_name(pid) for (pid,) in con.execute(
        f"SELECT photo_id FROM labels WHERE gate_category = '{codebook.KEEP_GATE}'").fetchall()}
    removed = 0
    for path in paths().dataset_images.glob("*.jpg"):
        if path.name not in kept:
            path.unlink()
            removed += 1
    return removed
