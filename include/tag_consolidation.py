"""
include/tag_consolidation.py
────────────────────────────
Reads all unique tags from the DuckDB photos table, maps them to a canonical
classification label, adds a canonical_label column, and prints a review
summary so you can adjust the mappings before training.

Usage
-----
    python include/tag_consolidation.py            # dry-run: print summary only
    python include/tag_consolidation.py --apply    # write canonical_label to DB

Review workflow
---------------
1. Run without --apply to see the full tag → label mapping and distribution.
2. Edit TAG_RULES below (add / move / rename entries as needed).
3. Run with --apply to commit the labels.
4. Re-run anytime — it overwrites canonical_label, so it is idempotent.
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Make include/ importable when run as a script ────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from include.db import get_conn  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# TAG RULES
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is (label, [keyword_fragments]).
# A photo's tags are matched against every fragment (case-insensitive substring).
# Rules are evaluated in order; the FIRST matching rule wins — so put the most
# specific labels at the top.
#
# Edit this table freely before running with --apply.
# ─────────────────────────────────────────────────────────────────────────────

TAG_RULES: list[tuple[str, list[str]]] = [
    # ── Specific object / phenomenon types (most specific first) ─────────────
    ("exoplanet", [
        "exoplanet", "planet", "transit", "wasp-", "hot jupiter",
        "protoplanet", "planetary system",
    ]),
    ("protostellar disk", [
        "protostellar", "protoplanetary", "debris disk", "circumstellar",
        "disk", "accretion",
    ]),
    ("deep field", [
        "deep field", "deepfield", "ultra deep", "background galaxies",
    ]),
    ("galaxy cluster", [
        "galaxy cluster", "cluster of galaxies", "gravitational lens",
        "lensing", "abell", "el gordo",
    ]),
    ("galaxy", [
        "galaxy", "galaxies", "spiral", "elliptical galaxy", "irregular galaxy",
        "interacting", "merger", "starburst", "quasar", "active galactic",
        "cartwheel", "stephan", "taffy", "arp ",
    ]),
    ("nebula", [
        "nebula", "nebulae", "pillars of creation", "carina", "orion",
        "ring nebula", "southern ring", "eta carinae", "lagoon",
        "tarantula", "bubble", "bow shock", "hii region",
        "emission nebula", "reflection nebula", "planetary nebula",
        "supernova remnant", "cassiopeia", "crab nebula",
    ]),
    ("star cluster", [
        "star cluster", "globular cluster", "open cluster", "stellar cluster",
        "pleiades", "praesepe", "westerlund",
    ]),
    ("star", [
        "star", "stellar", "brown dwarf", "white dwarf", "neutron star",
        "binary star", "variable star", "t tauri", "herbig ae",
        "wolf-rayet", "massive star",
    ]),
    ("cosmology", [
        "cosmology", "dark matter", "dark energy", "cosmic web",
        "large scale structure", "baryon", "reionization", "early universe",
        "big bang", "cmb", "hubble constant",
    ]),
    ("black hole", [
        "black hole", "event horizon", "accretion disk", "jet",
        "sgr a", "sagittarius a",
    ]),
    ("solar system", [
        "solar system", "jupiter", "saturn", "uranus", "neptune",
        "mars", "venus", "moon", "asteroid", "comet", "kuiper",
    ]),
    # ── Catch-all / administrative ────────────────────────────────────────────
    ("observatory / engineering", [
        "webb", "jwst", "telescope", "mirror", "instrument", "launch",
        "deployment", "commissioning", "nasa", "esa", "engineering",
    ]),
]

UNCLASSIFIED_LABEL = "unclassified"


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(tag: str) -> str:
    return tag.lower().strip()


def classify_tags(tags: list[str]) -> str:
    """
    Return the canonical label for a photo given its tag list.

    Evaluation order:
      1. For each rule (in TAG_RULES order), check whether ANY tag matches
         ANY keyword fragment for that rule.
      2. Return the label of the first matching rule.
      3. Return UNCLASSIFIED_LABEL if nothing matches.
    """
    normalised = [_normalise(t) for t in tags]
    for label, fragments in TAG_RULES:
        for fragment in fragments:
            frag = fragment.lower()
            if any(frag in tag for tag in normalised):
                return label
    return UNCLASSIFIED_LABEL


def load_photos(con) -> list[dict]:
    """Return all photos with their tag arrays."""
    rows = con.execute("SELECT photo_id, tags FROM photos").fetchall()
    return [{"photo_id": r[0], "tags": r[1] or []} for r in rows]


def build_mapping(photos: list[dict]) -> dict[str, str]:
    """Return {photo_id: canonical_label} for every photo."""
    return {p["photo_id"]: classify_tags(p["tags"]) for p in photos}


def build_tag_index(photos: list[dict]) -> dict[str, set[str]]:
    """Return {normalised_tag: {photo_id, ...}} for every tag that appears."""
    index: dict[str, set[str]] = defaultdict(set)
    for p in photos:
        for tag in p["tags"]:
            index[_normalise(tag)].add(p["photo_id"])
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(photos: list[dict], mapping: dict[str, str]) -> None:
    tag_index = build_tag_index(photos)

    # ── 1. All unique tags, sorted by frequency ───────────────────────────────
    tag_freq: Counter = Counter()
    for p in photos:
        for tag in p["tags"]:
            tag_freq[_normalise(tag)] += 1

    all_tags_sorted = sorted(tag_freq.items(), key=lambda x: -x[1])

    # ── 2. Which rule (if any) matched each tag ───────────────────────────────
    def first_rule_for_tag(tag: str) -> str:
        t = _normalise(tag)
        for label, fragments in TAG_RULES:
            if any(f.lower() in t for f in fragments):
                return label
        return UNCLASSIFIED_LABEL

    tag_to_rule: dict[str, str] = {t: first_rule_for_tag(t) for t, _ in all_tags_sorted}

    # ── 3. Per-label tag breakdown ────────────────────────────────────────────
    by_label: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for tag, freq in all_tags_sorted:
        by_label[tag_to_rule[tag]].append((tag, freq))

    label_counts = Counter(mapping.values())
    all_labels = [label for label, _ in TAG_RULES] + [UNCLASSIFIED_LABEL]

    print("\n" + "═" * 70)
    print("  TAG CONSOLIDATION REVIEW")
    print("═" * 70)
    print(f"  Photos: {len(photos)}   Unique tags: {len(all_tags_sorted)}\n")

    # ── Per-label section ─────────────────────────────────────────────────────
    for label in all_labels:
        count = label_counts.get(label, 0)
        tags_in_label = by_label.get(label, [])
        pct = 100 * count / len(photos) if photos else 0
        header = f"  [{label.upper()}]  {count} photos ({pct:.1f}%)"
        print(header)
        print("  " + "─" * (len(header) - 2))
        if tags_in_label:
            for tag, freq in tags_in_label:
                print(f"    {freq:>4}×  {tag}")
        else:
            print("    (no tags matched this label)")
        print()

    # ── Unmatched tags (not caught by any rule) ───────────────────────────────
    unmatched = [(t, f) for t, f in all_tags_sorted if tag_to_rule[t] == UNCLASSIFIED_LABEL]
    if unmatched:
        print("  UNMATCHED TAGS (consider adding to TAG_RULES):")
        print("  " + "─" * 46)
        for tag, freq in unmatched[:50]:
            print(f"    {freq:>4}×  {tag}")
        if len(unmatched) > 50:
            print(f"    … and {len(unmatched) - 50} more")
        print()

    # ── Label distribution table ──────────────────────────────────────────────
    print("  LABEL DISTRIBUTION")
    print("  " + "─" * 46)
    bar_max = 30
    max_count = max(label_counts.values()) if label_counts else 1
    for label in all_labels:
        count = label_counts.get(label, 0)
        pct = 100 * count / len(photos) if photos else 0
        bar_len = int(bar_max * count / max_count)
        bar = "█" * bar_len
        print(f"  {label:<28}  {count:>4}  {pct:>5.1f}%  {bar}")
    print()
    print("  Edit TAG_RULES in include/tag_consolidation.py, then re-run.")
    print("  When satisfied: python include/tag_consolidation.py --apply")
    print("═" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# DB write
# ─────────────────────────────────────────────────────────────────────────────

def ensure_canonical_label_column(con) -> None:
    """Add canonical_label column if it doesn't exist yet."""
    cols = {row[0] for row in con.execute("PRAGMA table_info('photos')").fetchall()}
    if "canonical_label" not in cols:
        con.execute("ALTER TABLE photos ADD COLUMN canonical_label TEXT")


def apply_labels(con, mapping: dict[str, str]) -> None:
    """Write canonical_label for every photo."""
    ensure_canonical_label_column(con)
    con.executemany(
        "UPDATE photos SET canonical_label = ? WHERE photo_id = ?",
        [(label, pid) for pid, label in mapping.items()],
    )
    print(f"  Applied labels to {len(mapping)} photos.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map Flickr tags to canonical JWST classification labels."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write canonical_label to DuckDB (default: dry-run / print only).",
    )
    args = parser.parse_args()

    con = get_conn(read_only=not args.apply)
    photos = load_photos(con)

    if not photos:
        print("No photos found in the database. Run the ingest DAG first.")
        con.close()
        return

    mapping = build_mapping(photos)
    print_summary(photos, mapping)

    if args.apply:
        apply_labels(con, mapping)
        con.close()
        print("Done. Re-run without --apply to verify the distribution.\n")
    else:
        con.close()
        print("Dry-run complete. No changes written to the database.\n")


if __name__ == "__main__":
    main()
