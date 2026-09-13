"""
03_review.py — build a local HTML sheet for hand-checking labels.

The model reports `confidence: high` on essentially everything, so there is no useful
low-confidence queue to review. Instead this draws a stratified random sample across
subject classes — over-weighting the small ones, where a single error moves the class
accuracy a lot — and renders image, label, rationale and caption side by side.

Output is a plain file you open in a browser. Nothing is uploaded; the thumbnails are
generated locally into review/thumbs/.

Usage:
    python build/03_review.py                 # 100 rows
    python build/03_review.py --n 60
    python build/03_review.py --include-rejected   # also sample gated-out rows

Mark anything wrong as you scroll, then hit "Copy corrections" at the bottom and paste
the result into review/corrections.json.
"""

import argparse
import html
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REVIEW = ROOT / "review"
THUMBS = REVIEW / "thumbs"
THUMB_PX = 420


def thumbnail(src: str, photo_id: str) -> str | None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    dest = THUMBS / f"{photo_id}.jpg"
    if dest.exists():
        return dest.name
    try:
        img = Image.open(src).convert("RGB")
        img.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
        img.save(dest, format="JPEG", quality=82)
        return dest.name
    except Exception as exc:
        print(f"  thumb failed {photo_id}: {exc}")
        return None


def stratified(rows: list[dict], n: int) -> list[dict]:
    """Sample n rows, spreading across subject classes rather than proportionally."""
    by_subject: dict[str, list] = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r)

    rng = random.Random(42)
    for v in by_subject.values():
        rng.shuffle(v)

    picked, i = [], 0
    # round-robin across classes so small classes are over-represented
    while len(picked) < n and any(len(v) > i for v in by_subject.values()):
        for k in sorted(by_subject):
            if len(by_subject[k]) > i and len(picked) < n:
                picked.append(by_subject[k][i])
        i += 1
    return picked


CSS = """
body{font:14px/1.5 system-ui,-apple-system,sans-serif;margin:0;background:#0f172a;color:#e2e8f0}
header{position:sticky;top:0;background:#0f172a;border-bottom:1px solid #1e293b;padding:14px 20px;z-index:10}
h1{margin:0;font-size:16px}
.sub{color:#94a3b8;font-size:12px;margin-top:3px}
.card{display:grid;grid-template-columns:440px 1fr;gap:20px;padding:20px;border-bottom:1px solid #1e293b}
.card.bad{background:#3f1d2b}
img{max-width:420px;border-radius:6px;background:#000}
.lab{font-size:13px}
.k{color:#64748b;display:inline-block;width:112px}
.v{color:#f1f5f9;font-weight:600}
.subj{font-size:19px;font-weight:700;color:#7dd3fc;margin-bottom:8px}
.cap{color:#94a3b8;font-size:12px;margin-top:10px;max-height:88px;overflow:auto;
     border-left:2px solid #334155;padding-left:9px}
.rat{color:#fbbf24;font-size:12px;margin-top:8px;font-style:italic}
.old{color:#f87171}
label.mark{display:inline-block;margin-top:12px;cursor:pointer;user-select:none;
     background:#1e293b;padding:6px 12px;border-radius:6px;font-size:12px}
input[type=text]{width:100%;margin-top:6px;background:#1e293b;border:1px solid #334155;
     color:#e2e8f0;padding:6px;border-radius:5px;font-size:12px}
footer{position:sticky;bottom:0;background:#0f172a;border-top:1px solid #1e293b;padding:14px 20px}
button{background:#0284c7;color:#fff;border:0;padding:9px 16px;border-radius:6px;
     font-size:13px;cursor:pointer}
#out{width:100%;height:90px;margin-top:9px;background:#020617;color:#7dd3fc;
     border:1px solid #334155;border-radius:6px;padding:8px;font:11px monospace;display:none}
a{color:#7dd3fc}
"""

JS = """
function mark(id,el){document.getElementById('c-'+id).classList.toggle('bad',el.checked);tally()}
function tally(){
  const n=document.querySelectorAll('.mk:checked').length;
  document.getElementById('n').textContent=n;
}
function dump(){
  const out=[];
  document.querySelectorAll('.mk:checked').forEach(cb=>{
    const id=cb.dataset.id;
    out.push({photo_id:id, was:cb.dataset.subj,
              should_be:document.getElementById('f-'+id).value.trim()||null});
  });
  const t=document.getElementById('out');
  t.style.display='block'; t.value=JSON.stringify(out,null,2); t.select();
  document.execCommand('copy');
}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--include-rejected", action="store_true")
    args = ap.parse_args()

    src = DATA / "jwst_space_images.parquet"
    if not src.exists():
        sys.exit("run build/02_assemble.py first")

    con = duckdb.connect()
    cols = "photo_id,title,description,subject,modality,object_name,object_name_verified,\
instrument,confidence,rationale,source_tag_label,source_image_path,flickr_url"
    rows = [dict(zip(cols.replace("\\", "").split(","), r))
            for r in con.execute(f"SELECT {cols} FROM '{src}'").fetchall()]
    print(f"{len(rows)} kept rows available")

    sample = stratified(rows, args.n)
    print(f"sampled {len(sample)} across {len({r['subject'] for r in sample})} classes")

    REVIEW.mkdir(exist_ok=True)
    THUMBS.mkdir(exist_ok=True)

    cards = []
    for r in sample:
        thumb = thumbnail(r["source_image_path"], r["photo_id"])
        img = f'<img src="thumbs/{thumb}" loading="lazy">' if thumb else "<div>no image</div>"
        pid = r["photo_id"]
        old = r["source_tag_label"]
        old_html = (
            f'<div><span class="k">old tag label</span>'
            f'<span class="old">{html.escape(str(old))}</span></div>' if old else ""
        )
        vflag = "" if r["object_name_verified"] else ' <span class="old">(unverified)</span>'
        cards.append(f"""
<div class="card" id="c-{pid}">
  <div>{img}</div>
  <div class="lab">
    <div class="subj">{html.escape(r['subject'])}</div>
    <div><span class="k">modality</span><span class="v">{html.escape(r['modality'])}</span></div>
    <div><span class="k">object</span><span class="v">{html.escape(str(r['object_name']))}</span>{vflag}</div>
    <div><span class="k">instrument</span><span class="v">{html.escape(r['instrument'])}</span></div>
    {old_html}
    <div class="rat">{html.escape(r['rationale'])}</div>
    <div class="cap"><b>{html.escape(r['title'])}</b><br>{html.escape((r['description'] or '')[:420])}</div>
    <div><a href="{r['flickr_url']}" target="_blank">flickr ↗</a></div>
    <label class="mark"><input type="checkbox" class="mk" data-id="{pid}"
      data-subj="{html.escape(r['subject'])}" onchange="mark('{pid}',this)"> wrong</label>
    <input type="text" id="f-{pid}" placeholder="correct label (optional)">
  </div>
</div>""")

    page = f"""<!doctype html><meta charset="utf-8"><title>JWST label review</title>
<style>{CSS}</style>
<header><h1>JWST golden dataset — label review</h1>
<div class="sub">{len(sample)} of {len(rows)} kept rows, stratified across subject classes ·
marked wrong: <b id="n">0</b></div></header>
{''.join(cards)}
<footer><button onclick="dump()">Copy corrections</button>
<textarea id="out"></textarea></footer>
<script>{JS}</script>"""

    out = REVIEW / "review.html"
    out.write_text(page)
    print(f"\nwrote {out}")
    print(f"open it with:  open {out}")


if __name__ == "__main__":
    main()
