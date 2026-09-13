"""
compare_versions.py — diff two label runs, field by field.

Codebook revisions are the main lever on quality in this project, so it matters to
see exactly what a revision changed rather than trusting that it helped. Compares a
backup JSONL against the current one over the photo_ids they share.

Usage:
    python build/compare_versions.py
    python build/compare_versions.py --old data/raw_labels.v2.bak.jsonl --examples 8
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
FIELDS = ["gate_category", "subject", "modality", "object_name", "instrument"]


def load(path: Path) -> dict[str, dict]:
    return {
        r["photo_id"]: r
        for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="raw_labels.v2.bak.jsonl")
    ap.add_argument("--new", default="raw_labels.jsonl")
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()

    old, new = load(DATA / args.old), load(DATA / args.new)
    shared = sorted(set(old) & set(new))
    print(f"{len(old)} old rows, {len(new)} new, {len(shared)} shared\n")

    # only rows whose codebook version actually moved are meaningful
    moved = [p for p in shared
             if old[p].get("codebook_version") != new[p].get("codebook_version")]
    print(f"{len(moved)} rows changed codebook version "
          f"({old[moved[0]].get('codebook_version')} -> {new[moved[0]].get('codebook_version')})\n"
          if moved else "no version change\n")

    scope = moved or shared
    for f in FIELDS:
        changed = [p for p in scope if old[p][f] != new[p][f]]
        pct = 100 * len(changed) / max(len(scope), 1)
        print(f"{f:<16} {len(changed):>4}/{len(scope)} changed  ({pct:.0f}%)")
        if not changed:
            continue
        transitions = Counter((old[p][f], new[p][f]) for p in changed)
        for (a, b), n in transitions.most_common(6):
            print(f"      {str(a):<24} -> {str(b):<24} {n:>4}")
        print()

    # distribution shift, which is easier to read than a transition table
    print("=" * 62)
    for f in ("modality", "instrument"):
        a = Counter(old[p][f] for p in scope)
        b = Counter(new[p][f] for p in scope)
        print(f"\n{f}:")
        for k in sorted(set(a) | set(b), key=lambda k: -b.get(k, 0)):
            d = b.get(k, 0) - a.get(k, 0)
            arrow = f"  {d:+d}" if d else ""
            print(f"   {k:<24} {a.get(k,0):>4} -> {b.get(k,0):>4}{arrow}")

    if args.examples:
        print("\n" + "=" * 62)
        print(f"\nsample of changed rows:")
        shown = 0
        for p in scope:
            diff = [f for f in FIELDS if old[p][f] != new[p][f]]
            if not diff or shown >= args.examples:
                continue
            print(f"\n  {new[p]['title'][:62]}")
            for f in diff:
                print(f"      {f:<14} {str(old[p][f])[:26]:<26} -> {new[p][f]}")
            shown += 1


if __name__ == "__main__":
    main()
