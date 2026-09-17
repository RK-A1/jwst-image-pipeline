"""
Build the animated gallery shown at the top of the README.

Frames are taken from the published dataset, and each caption is drawn from that
photo's own row, so the gallery shows what the pipeline produced rather than a
hand-written list. Animated WebP keeps full colour, which a 256-colour GIF cannot on
images that are mostly smooth gradients and star fields.

    python scripts/build_gallery.py

The images are NASA/ESA/CSA/STScI press releases; the dataset carries the credit.
"""

import argparse
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from include.jwst_pipeline.config import paths  # noqa: E402
from include.jwst_pipeline.images import open_rgb  # noqa: E402

# Recognisable observations, spread across subject classes.
FRAMES = [
    "52211883799",  # Cosmic Cliffs, Carina Nebula
    "53876176351",  # Pillars of Creation
    "52338778943",  # Tarantula Nebula
    "53686724794",  # Horsehead Nebula
    "52799298357",  # Cassiopeia A
    "52259483705",  # Cartwheel Galaxy
    "54163774891",  # Sombrero Galaxy
    "53268962997",  # Jupiter
    "54692963181",  # JADES deep field
]

# The dataset copies catalogue names verbatim, which is right for the data and cryptic
# in a caption. These few are shown under the name a reader will recognise.
DISPLAY_NAMES = {
    "54163774891": "Sombrero Galaxy",
    "54692963181": "JADES deep field",
    "52211883799": "Cosmic Cliffs, Carina Nebula",
}

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "gallery.webp"
WIDTH, HEIGHT = 1200, 675          # 16:9, wide enough for a README hero
MS_PER_FRAME = 2200
MARGIN = 28
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size: int):
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _frame(image_path: Path, title: str, subject: str, instrument: str):
    """One letterboxed frame with its dataset labels drawn in the corner."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    photo = open_rgb(image_path)
    photo.thumbnail((WIDTH, HEIGHT), Image.LANCZOS)
    canvas.paste(photo, ((WIDTH - photo.width) // 2, (HEIGHT - photo.height) // 2))

    draw = ImageDraw.Draw(canvas, "RGBA")
    label = f"{subject}   ·   {instrument}"
    title_font, label_font = _font(30), _font(19)

    # A dark strip behind the text, so it stays readable over a bright nebula.
    text_top = HEIGHT - MARGIN - 62
    draw.rectangle([0, text_top - 14, WIDTH, HEIGHT], fill=(0, 0, 0, 130))
    draw.text((MARGIN, text_top), title, font=title_font, fill=(255, 255, 255))
    draw.text((MARGIN, text_top + 38), label, font=label_font, fill=(125, 211, 252))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    dataset = paths().dataset / "jwst_space_images.parquet"
    con = duckdb.connect()
    rows = {
        r[0]: r[1:] for r in con.execute(f"""
            SELECT photo_id,
                   coalesce(object_name, regexp_replace(title, '\\s*\\(.*', '')) AS name,
                   subject, instrument, image_file
            FROM '{dataset}' WHERE photo_id IN ({','.join("'" + f + "'" for f in FRAMES)})
        """).fetchall()
    }
    missing = [f for f in FRAMES if f not in rows]
    if missing:
        sys.exit(f"not in the dataset: {missing}")

    frames = []
    for photo_id in FRAMES:
        name, subject, instrument, image_file = rows[photo_id]
        name = DISPLAY_NAMES.get(photo_id, name)
        path = paths().dataset / image_file
        if not path.exists():
            sys.exit(f"no image for {photo_id} at {path}; run the jwst_dataset DAG first")
        frames.append(_frame(path, name, subject, instrument))
        print(f"  {photo_id}  {name}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=MS_PER_FRAME, loop=0, quality=72, method=6)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
