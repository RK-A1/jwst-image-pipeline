"""
02_assemble.py — turn raw model output into the finished dataset.

Joins data/raw_labels.jsonl back onto data/candidates.parquet, validates the
extracted object names, splits gated-out rows off into their own file, and
optionally writes downscaled copies of the kept images.

Usage:
    python build/02_assemble.py
    python build/02_assemble.py --labels bench_12b.jsonl   # assemble a benchmark run
    python build/02_assemble.py --images                   # also downscale images/
"""

import argparse
import json
import re
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGES = ROOT / "images"
IMAGE_LONG_EDGE = 1024


def normalise(s: str) -> str:
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
    # "Messier 51 (M51)" -> "Messier 51" before matching; keep the gloss if stripping
    # it would leave nothing.
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
    if letter:                       # exoplanet designations: WASP-96 b
        out += f" {letter.lower()}"
    return out


_QUALIFIER = re.compile(r"\s*[\(\[][^)\]]*[)\]]\s*")


def release_key(title: str) -> str:
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
    try:
        import numpy as np
    except ImportError:
        return {}
    M = np.asarray(vecs, dtype=np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    S = M @ M.T
    np.fill_diagonal(S, 0.0)

    n = len(ids)
    cluster = [-1] * n
    cid = 0
    for i in range(n):
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


def write_images(rows: list[tuple[str, str]]) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        sys.exit("--images needs Pillow: pip install Pillow")
    Image.MAX_IMAGE_PIXELS = None  # JWST originals exceed the decompression-bomb limit

    IMAGES.mkdir(parents=True, exist_ok=True)
    written = failed = 0
    for i, (photo_id, src) in enumerate(rows, 1):
        dest = IMAGES / f"{photo_id}.jpg"
        if dest.exists():
            written += 1
            continue
        try:
            img = Image.open(src).convert("RGB")
            img.thumbnail((IMAGE_LONG_EDGE, IMAGE_LONG_EDGE), Image.LANCZOS)
            img.save(dest, format="JPEG", quality=88)
            written += 1
        except Exception as exc:
            print(f"  image failed for {photo_id}: {exc}")
            failed += 1
        if i % 100 == 0:
            print(f"  images: {i}/{len(rows)}")
    return written, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="raw_labels.jsonl")
    ap.add_argument("--images", action="store_true", help="write downscaled copies")
    args = ap.parse_args()

    labels_path = DATA / args.labels
    cand_path = DATA / "candidates.parquet"
    for p in (labels_path, cand_path):
        if not p.exists():
            sys.exit(f"missing {p}")

    con = duckdb.connect()
    con.execute(f"CREATE VIEW cand AS SELECT * FROM '{cand_path}'")
    con.execute(f"CREATE VIEW lab AS SELECT * FROM read_json_auto('{labels_path}')")

    n_lab = con.execute("SELECT count(*) FROM lab").fetchone()[0]
    n_cand = con.execute("SELECT count(*) FROM cand").fetchone()[0]
    print(f"candidates: {n_cand}   labelled: {n_lab}")

    # Any gate value outside the enum would otherwise be counted as a rejection and
    # silently drop a real observation — that happened to 5 rows before the tool
    # definition was switched to strict mode. Fail loudly instead.
    VALID_GATES = (
        "'astronomical_observation','hardware_engineering','people_event',"
        "'artwork_illustration','promotional_other'"
    )
    invalid = con.execute(
        f"SELECT gate_category, count(*) FROM lab "
        f"WHERE gate_category NOT IN ({VALID_GATES}) GROUP BY 1"
    ).fetchall()
    if invalid:
        print("\n!! INVALID gate_category values — these are NOT reliable rejections:")
        for g, n in invalid:
            print(f"     {g!r}: {n} rows")
        print("   Re-label them; do not trust this build.\n")

    # Verify object_name appears verbatim in the source text. Small models mangle
    # catalogue designations, and this is the field where that does the most damage.
    joined = con.execute("""
        SELECT c.photo_id, c.title, c.description, l.object_name
        FROM cand c JOIN lab l USING (photo_id)
        WHERE l.object_name IS NOT NULL
    """).fetchall()
    checked = [
        (pid, normalise(name) in normalise(f"{title} {desc}"), canonical_name(name))
        for pid, title, desc, name in joined
    ]
    con.execute(
        "CREATE TABLE verif (photo_id VARCHAR, object_name_verified BOOLEAN, "
        "object_name_normalized VARCHAR)"
    )
    if checked:
        con.executemany("INSERT INTO verif VALUES (?, ?, ?)", checked)

    # ── variant grouping ────────────────────────────────────────────────────
    # Three levels, because each misses what the others catch:
    #   object_name_normalized : every image of this object          (coarsest)
    #   release_group          : one press release / figure set      (the useful one)
    #   near_duplicate_group   : literally the same pixels           (rare, 2%)
    gate_ok = "l.gate_category = 'astronomical_observation'"
    grp_rows = con.execute(f"""
        SELECT c.photo_id, c.title FROM cand c JOIN lab l USING (photo_id)
        WHERE {gate_ok} ORDER BY c.photo_id
    """).fetchall()
    rel = {pid: release_key(t) for pid, t in grp_rows}

    emb_path = DATA / "embeddings.parquet"
    dup: dict[str, int] = {}
    if emb_path.exists():
        e = con.execute(f"""
            SELECT photo_id, embedding FROM '{emb_path}'
            WHERE photo_id IN (SELECT c.photo_id FROM cand c JOIN lab l USING (photo_id)
                               WHERE {gate_ok})
            ORDER BY photo_id
        """).fetchall()
        if e:
            dup = duplicate_clusters([r[0] for r in e], [r[1] for r in e])

    # one primary per release group: prefer a clean image over an annotated or
    # multi-panel one, then the largest file (usually the highest resolution)
    RANK = {"image": 0, "annotated_image": 1, "comparison_composite": 2, "spectrum_or_plot": 3}
    meta = con.execute(f"""
        SELECT c.photo_id, l.modality, c.source_image_path
        FROM cand c JOIN lab l USING (photo_id) WHERE {gate_ok}
    """).fetchall()
    by_rel: dict[str, list] = {}
    for pid, modality, path in meta:
        size = Path(path).stat().st_size if path and Path(path).exists() else 0
        by_rel.setdefault(rel[pid], []).append((RANK.get(modality, 9), -size, pid))
    primary = {sorted(v)[0][2] for v in by_rel.values()}

    con.execute("CREATE TABLE grouping (photo_id VARCHAR, release_group VARCHAR, "
                "near_duplicate_group INTEGER, is_primary BOOLEAN)")
    con.executemany("INSERT INTO grouping VALUES (?,?,?,?)",
                    [(pid, rel[pid], dup.get(pid), pid in primary) for pid, _ in grp_rows])
    print(f"\ngrouping: {len(set(rel.values()))} release groups, "
          f"{len(primary)} primaries, "
          f"{len(grp_rows)-len(set(dup.values())) if dup else 0} pixel-level near-duplicates")

    con.execute("""
        CREATE VIEW labelled AS
        SELECT
            c.photo_id, c.title, c.description,
            l.subject, l.modality, l.object_name,
            v.object_name_normalized,
            coalesce(v.object_name_verified, l.object_name IS NULL) AS object_name_verified,
            l.instrument, l.confidence, l.rationale, l.gate_category,
            g.release_group, g.near_duplicate_group, g.is_primary,
            c.flickr_tags, c.date_taken, c.flickr_url,
            c.source_tag_label, c.source_image_path, c.has_embedding,
            l.model, l.codebook_version, l.vision AS labelled_with_vision
        FROM cand c
        JOIN lab l USING (photo_id)
        LEFT JOIN verif v USING (photo_id)
        LEFT JOIN grouping g USING (photo_id)
    """)

    kept_cols = """photo_id, title, description, subject, modality, object_name,
        object_name_normalized, object_name_verified, instrument, confidence, rationale,
        release_group, near_duplicate_group, is_primary, flickr_tags,
        date_taken, flickr_url, source_tag_label, source_image_path, has_embedding,
        model, codebook_version, labelled_with_vision"""

    con.execute(f"""
        COPY (SELECT {kept_cols} FROM labelled
              WHERE gate_category = 'astronomical_observation'
              ORDER BY photo_id)
        TO '{DATA / "jwst_space_images.parquet"}' (FORMAT parquet)
    """)
    con.execute(f"""
        COPY (SELECT photo_id, title, gate_category AS rejected_as, rationale,
                     date_taken, flickr_url, source_tag_label
              FROM labelled WHERE gate_category <> 'astronomical_observation'
              ORDER BY photo_id)
        TO '{DATA / "rejected.parquet"}' (FORMAT parquet)
    """)

    kept = con.execute(
        "SELECT count(*) FROM labelled WHERE gate_category = 'astronomical_observation'"
    ).fetchone()[0]
    print(f"\nkept {kept}   rejected {n_lab - kept}")

    print("\nrejected by category:")
    for cat, n in con.execute("""
        SELECT gate_category, count(*) FROM labelled
        WHERE gate_category <> 'astronomical_observation'
        GROUP BY 1 ORDER BY 2 DESC
    """).fetchall():
        print(f"  {cat:<26} {n:>4}")

    print("\nsubject distribution:")
    for s, n in con.execute("""
        SELECT subject, count(*) FROM labelled
        WHERE gate_category = 'astronomical_observation'
        GROUP BY 1 ORDER BY 2 DESC
    """).fetchall():
        print(f"  {s:<26} {n:>4}")

    print("\nmodality:")
    for m, n in con.execute("""
        SELECT modality, count(*) FROM labelled
        WHERE gate_category = 'astronomical_observation'
        GROUP BY 1 ORDER BY 2 DESC
    """).fetchall():
        print(f"  {m:<26} {n:>4}")

    unver, named = con.execute("""
        SELECT count(*) FILTER (WHERE NOT object_name_verified),
               count(*) FILTER (WHERE object_name IS NOT NULL)
        FROM labelled WHERE gate_category = 'astronomical_observation'
    """).fetchone()
    low = con.execute("""
        SELECT count(*) FROM labelled
        WHERE gate_category = 'astronomical_observation' AND confidence = 'low'
    """).fetchone()[0]
    print(f"\nreview queue:")
    print(f"  object_name not found verbatim in caption: {unver} of {named} named")
    print(f"  confidence = low:                          {low}")

    # How often the new label disagrees with the old keyword-derived one.
    print("\nvs. the old tag-based label (where one existed):")
    for old, new, n in con.execute("""
        SELECT source_tag_label, subject, count(*) n FROM labelled
        WHERE gate_category = 'astronomical_observation' AND source_tag_label IS NOT NULL
        GROUP BY 1, 2 HAVING count(*) >= 2 ORDER BY n DESC LIMIT 12
    """).fetchall():
        flag = "" if old == new else "  <- changed"
        print(f"  {str(old):<28} -> {new:<22} {n:>4}{flag}")

    if args.images:
        rows = con.execute("""
            SELECT photo_id, source_image_path FROM labelled
            WHERE gate_category = 'astronomical_observation' ORDER BY photo_id
        """).fetchall()
        print(f"\nwriting {len(rows)} downscaled images...")
        w, f = write_images(rows)
        print(f"  {w} written, {f} failed")

    # CSV alongside the parquet, for eyeballing in a spreadsheet. Short, useful
    # columns first; newlines collapsed so one row is one line; tag lists flattened.
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
        COPY (SELECT {csv_cols} FROM labelled
              WHERE gate_category = 'astronomical_observation' ORDER BY subject, photo_id)
        TO '{DATA / "jwst_space_images.csv"}' (HEADER, DELIMITER ',')
    """)
    con.execute(f"""
        COPY (SELECT photo_id, gate_category AS rejected_as, title, rationale,
                     flickr_url, date_taken, source_tag_label
              FROM labelled WHERE gate_category <> 'astronomical_observation'
              ORDER BY gate_category, photo_id)
        TO '{DATA / "rejected.csv"}' (HEADER, DELIMITER ',')
    """)

    for f in ("jwst_space_images.parquet", "rejected.parquet",
              "jwst_space_images.csv", "rejected.csv"):
        print(f"wrote data/{f}  ({(DATA / f).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
