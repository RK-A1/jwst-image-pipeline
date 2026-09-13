"""
01_label.py — label candidates with either a local Gemma 4 model or the Claude API.

One call per photo does the gate and the labelling together. Decoding is constrained
against SCHEMA, so responses are always structurally valid; the codebook is the
system prompt.

Resumable: every response is appended to data/raw_labels.jsonl as it lands, and
photo_ids already present are skipped. Safe to interrupt and restart.

Backends:
    --backend ollama   local, free, slow      (default model gemma4:12b)
    --backend api      Claude, paid, fast     (default model claude-sonnet-5)

The API key is read from ANTHROPIC_API_KEY or from .env at the project root.

Usage:
    python build/01_label.py --backend api --vision
    python build/01_label.py --backend api --model claude-haiku-4-5 --limit 20
    python build/01_label.py --ids 5368,5381 --out bench.jsonl
"""

import argparse
import base64
import html as html_mod
import re
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OLLAMA = "http://127.0.0.1:11434/api/chat"

DEFAULT_MODEL = "gemma4:12b"          # ollama backend
DEFAULT_API_MODEL = "claude-sonnet-5"  # api backend
DESC_CHARS = 4000        # p99 caption is 3737 chars; covers the alt-text block
IMAGE_LONG_EDGE = 896    # downscale before sending; JWST originals are up to 100 MP

GATE_CATEGORIES = [
    "astronomical_observation", "hardware_engineering", "people_event",
    "artwork_illustration", "promotional_other",
]
SUBJECTS = [
    "galaxy", "galaxy_cluster", "deep_field", "nebula", "planetary_nebula",
    "supernova_remnant", "star", "star_cluster", "protoplanetary_disk",
    "exoplanet", "solar_system", "black_hole", "other_astronomical", "not_applicable",
]
MODALITIES = [
    "image", "annotated_image", "spectrum_or_plot", "comparison_composite",
    "not_applicable",
]
INSTRUMENTS = [
    "NIRCam", "MIRI", "NIRSpec", "NIRISS", "FGS", "multiple", "non-Webb", "unknown",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "gate_category": {"type": "string", "enum": GATE_CATEGORIES},
        "subject": {"type": "string", "enum": SUBJECTS},
        "modality": {"type": "string", "enum": MODALITIES},
        "object_name": {"type": ["string", "null"]},
        "instrument": {"type": "string", "enum": INSTRUMENTS},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "rationale": {"type": "string"},
    },
    "required": [
        "gate_category", "subject", "modality", "object_name",
        "instrument", "confidence", "rationale",
    ],
    # Required for strict tool use, which is what actually enforces the enums.
    # Without it the schema is advisory and the model can emit out-of-enum values —
    # it put a modality value into gate_category on 5 records in the first run.
    "additionalProperties": False,
}

SYSTEM = """You classify photos from the NASA Webb Telescope Flickr account using their title, caption, and (when provided) the image itself.

STEP 1 — GATE. Is this actual observational data of something in space?
  astronomical_observation : data captured by a telescope of an object beyond Earth. Images, mosaics, spectra, light curves, and multi-observatory composites all count.
  hardware_engineering     : the telescope or its parts — mirrors, sunshield, instruments, clean rooms, test rigs, launch vehicles, ground stations.
  people_event             : humans as the subject — ceremonies, briefings, conferences, museum exhibits, portraits, outreach events.
  artwork_illustration     : artist's concepts, renderings, #JWSTArt community submissions, infographics carrying no observational data.
  promotional_other        : logos, posters, anniversary graphics, mission patches, broadcast stills, anything else.

Gate rules:
  - Data plotted on axes is still an observation. A transmission spectrum is astronomical_observation.
  - An artist's concept is never an observation, even of a real object Webb studied. Ask whether photons from the object made the picture.
  - TEXT PRINTED IN THE IMAGE OVERRIDES THE CAPTION. NASA marks renderings with "Artist's Concept", "Illustration", "Simulation", or "Artist's Impression", usually small and in a corner. If you can see that text, the answer is artwork_illustration no matter how much real science the caption describes — and it usually describes a lot, because the release is about real results. This is the one case where the caption misleads and only the image is reliable.
  - Composites including Chandra, Hubble, Euclid, Spitzer, or ground-based data still count.
  - When the caption describes an event but the image shows an observation, judge the image.

If the gate is not astronomical_observation, set subject and modality to "not_applicable", object_name to null, instrument to "unknown", and stop.

STEP 2 — SUBJECT. The primary astronomical object. Choose exactly one.
  galaxy              : one galaxy or a small interacting group — spiral, elliptical, irregular, starburst.
  galaxy_cluster      : a bound cluster, including lensing clusters (Abell, SMACS, El Gordo, Bullet Cluster).
  deep_field          : a wide survey field whose subject is the background galaxy population (JADES, CEERS, COSMOS-Web).
  nebula              : interstellar gas and dust — emission, reflection, dark nebulae, HII regions, star-forming regions, molecular clouds.
  planetary_nebula    : the ejected envelope of a dying low-mass star (Ring, Southern Ring, NGC 3132).
  supernova_remnant   : debris of an exploded star (Cassiopeia A, Crab, SN 1987A).
  star                : an individual star or stellar system — binaries, brown dwarfs, white dwarfs, Wolf-Rayet, variables.
  star_cluster        : a globular or open cluster.
  protoplanetary_disk : circumstellar and debris disks, protostars, planet-forming regions, Herbig-Haro objects, protostellar jets.
  exoplanet           : a planet outside the solar system, or its atmosphere.
  solar_system        : planets, moons, rings, asteroids, comets, Kuiper Belt objects.
  black_hole          : black holes and their surroundings — AGN, quasars, Sgr A*, relativistic jets.
  other_astronomical  : a real observation fitting none of the above. Use sparingly.

Subject rules — these decide the cases that are most often got wrong:
  - "Planetary nebula" has nothing to do with planets. It is planetary_nebula, never exoplanet.
  - A protoplanetary or debris disk is protoplanetary_disk, never exoplanet. Only use exoplanet when a planet itself is the subject.
  - supernova_remnant outranks nebula. planetary_nebula outranks nebula.
  - deep_field vs galaxy_cluster: decide by what the caption is ABOUT, not by what is named. deep_field when the subject is the population of distant/background galaxies, the depth of the exposure, or the early universe — even when a foreground lensing cluster is named. galaxy_cluster when the subject is the cluster itself: its mass, structure, lensing, or member galaxies. "Deepest infrared image of the universe yet" is deep_field even though it shows SMACS 0723; "Webb pierces the Bullet Cluster, refines its mass" is galaxy_cluster.
  - A patch inside a galaxy is nebula, not galaxy, when the subject is the interstellar material — dust lanes, star fields, HII regions — even if the host galaxy is named. Use galaxy only when the galaxy as a whole is the subject: its morphology, structure, or identity. A MIRI test frame showing dust and stars in part of the Large Magellanic Cloud is nebula.
  - Sgr A* / Sagittarius A* is black_hole, not star.
  - A star-forming region is nebula unless one protostar or disk is the subject.
  - A galaxy is black_hole only when the caption is about its AGN, jet, or central engine.
  - Comets and asteroids are solar_system, never star.

STEP 3 — MODALITY.
  image                : a direct image or mosaic, unannotated.
  annotated_image      : an image with graphics ADDED ON TOP of the observation. Counts as annotated: text or object labels; arrows, crosshairs, tick marks; circles, ellipses, dashed field-of-view boxes; contour lines; region markers; inset boxes and the callout lines joining them; compass roses; scale bars; colour/filter keys ("blue = F090W"); colour bars; coordinate grids; panel letters; a star glyph or black occulting disc marking a masked host star in coronagraphic exoplanet images; a text strip added above or below the frame. If a graphic was drawn over the pixels it is annotated, even when the marks carry no text.
                         NOT annotation — these are the data itself: diffraction spikes (the six-pointed pattern from Webb's mirrors); ragged or stepped edges (the detector mosaic footprint); black background or letterboxing; false colour (all these images are false colour); noise and detector artifacts.
                         Clean and annotated versions of the same frame are often published as a pair and may differ only by a few drawn lines. Look closely before choosing image.
  spectrum_or_plot     : data on axes — spectra, light curves, transmission curves.
  comparison_composite : TWO OR MORE SEPARATE PANELS presented together in one graphic — Webb beside Hubble, NIRCam beside MIRI, before/after, a strip of framed insets. The test is whether you can see distinct panel boundaries. A single picture that BLENDS data from several telescopes into one frame is NOT this — "Chandra Adds X-ray Vision to Webb Images" is one merged picture, so it is image. NASA says "composite" to mean blended; here it means panelled.
If both annotated and a comparison, choose comparison_composite.
MODALITY IS DECIDED BY THE IMAGE, NOT THE CAPTION. Count the panels you can actually see in the picture. IGNORE the words "composite", "combined", "merged", "blended" and "multi-observatory" wherever they appear in the text — they describe how the DATA was processed, not how the GRAPHIC is laid out. Webb and Chandra data merged into a single frame is a single frame: that is image, not comparison_composite. Only visible panel divisions — separate sub-pictures with their own borders — make it comparison_composite.

STEP 4 — OBJECT NAME. The catalogue designation or proper name of the primary object, copied VERBATIM from the title or caption — "NGC 6334", "WASP-96 b", "SMACS 0723", "Cassiopeia A". Copy exactly; do not normalise, expand, or correct. Prefer the catalogue designation when both appear. Use null when no specific object is named. Never invent one.

STEP 5 — INSTRUMENT. NIRCam, MIRI, NIRSpec, NIRISS, FGS, multiple, non-Webb, or unknown.
  - unknown IS THE CORRECT ANSWER whenever the text does not name an instrument. It is not a failure or a hedge, and most captions simply never say. Never infer an instrument from the subject, from how the image looks, or from the kind of observation.
  - multiple requires TWO OR MORE Webb instruments named explicitly. One named instrument means that instrument, not multiple.
  - The instrument is usually in the title in parentheses: "(NIRCam Image)", "(NIRSpec MSA Emission Spectra)", "(NIRCam and MIRI)".
  - non-Webb when the data is entirely from another observatory. For a Webb + Chandra composite, name the Webb instrument if stated, else unknown.

STEP 6 — CONFIDENCE and RATIONALE. Use low when the caption is thin or the case is genuinely contested, not merely when the object is unfamiliar. Keep the rationale under 20 words.

Respond only with the JSON object."""


def downscale_b64(path: str) -> str | None:
    """Return a base64 JPEG downscaled to IMAGE_LONG_EDGE, or None if unreadable."""
    try:
        from PIL import Image
    except ImportError:
        sys.exit("--vision needs Pillow: pip install Pillow")
    Image.MAX_IMAGE_PIXELS = None  # JWST originals exceed the decompression-bomb limit
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    img.thumbnail((IMAGE_LONG_EDGE, IMAGE_LONG_EDGE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


_TAG_RE = re.compile(r"<[^>]+>")


def user_text(title: str, desc: str) -> str:
    """
    Strip HTML and decode entities BEFORE truncating. Flickr stores raw markup, and a
    single `<a href="..." rel="noreferrer nofollow">` burns ~75 characters of budget
    carrying no meaning.

    Truncation matters more than it looks: 45% of these captions end with a NASA/ESA
    "Image description:" accessibility block describing the visible content, and it
    sits at the very end, where a short window cuts it off.
    """
    clean = html_mod.unescape(_TAG_RE.sub("", desc or "")).strip()
    return f"TITLE: {title}\n\nCAPTION: {clean[:DESC_CHARS]}"


def call_ollama(model: str, title: str, desc: str, image_b64: str | None) -> tuple[dict, dict]:
    """Local inference. Decoding is constrained by SCHEMA via Ollama's `format`."""
    msg = {"role": "user", "content": user_text(title, desc)}
    if image_b64:
        msg["images"] = [image_b64]

    body = json.dumps({
        "model": model,
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0, "num_ctx": 4096},
        "messages": [{"role": "system", "content": SYSTEM}, msg],
    }).encode()

    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return json.loads(raw["message"]["content"]), raw


def load_dotenv() -> None:
    """Read KEY=VALUE lines from .env into the environment, without clobbering."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_client = None


def api_client():
    global _client
    if _client is None:
        try:
            import anthropic
        except ImportError:
            sys.exit("--backend api needs the SDK: pip install anthropic")
        load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit(
                "ANTHROPIC_API_KEY is not set.\n"
                f"  export it, or put it in {ROOT / '.env'} (gitignored)."
            )
        _client = anthropic.Anthropic()
    return _client


def call_api(model: str, title: str, desc: str, image_b64: str | None) -> tuple[dict, dict]:
    """
    Hosted inference. The schema is enforced by handing SCHEMA to a single tool and
    forcing that tool, so the response is always a valid object rather than prose to
    parse. The codebook is cached — it is byte-identical on every call and dwarfs the
    per-photo content.
    """
    client = api_client()

    content: list[dict] = []
    if image_b64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
        })
    content.append({"type": "text", "text": user_text(title, desc)})

    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[{
            "name": "label_photo",
            "description": "Record the classification for this photo.",
            "input_schema": SCHEMA,
            "strict": True,
        }],
        tool_choice={"type": "tool", "name": "label_photo"},
        messages=[{"role": "user", "content": content}],
    )

    for block in resp.content:
        if block.type == "tool_use":
            u = resp.usage
            return dict(block.input), {
                "prompt_eval_count": u.input_tokens,
                "eval_count": u.output_tokens,
                "cache_read": getattr(u, "cache_read_input_tokens", None),
                "cache_write": getattr(u, "cache_creation_input_tokens", None),
            }
    raise ValueError(f"no tool_use block in response (stop_reason={resp.stop_reason})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "api"], default="ollama")
    ap.add_argument("--model", default=None,
                    help="default: gemma4:12b for ollama, claude-sonnet-5 for api")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", help="comma-separated photo_ids")
    ap.add_argument("--vision", action="store_true", help="also send the image")
    ap.add_argument("--out", default="raw_labels.jsonl")
    args = ap.parse_args()
    if args.model is None:
        args.model = DEFAULT_MODEL if args.backend == "ollama" else DEFAULT_API_MODEL
    call = call_ollama if args.backend == "ollama" else call_api

    src = DATA / "candidates.parquet"
    if not src.exists():
        sys.exit("data/candidates.parquet missing — run build/00_extract.py first")

    con = duckdb.connect()
    q = f"SELECT photo_id, title, description, source_image_path FROM '{src}'"
    if args.ids:
        ids = ",".join(f"'{i.strip()}'" for i in args.ids.split(","))
        q += f" WHERE photo_id IN ({ids})"
    q += " ORDER BY photo_id"
    if args.limit:
        q += f" LIMIT {args.limit}"
    rows = con.execute(q).fetchall()

    out_path = DATA / args.out
    done = set()
    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["photo_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [r for r in rows if r[0] not in done]

    print(f"backend={args.backend} model={args.model} vision={args.vision}")
    print(f"{len(rows)} selected, {len(done)} already done, {len(todo)} to go\n")
    if not todo:
        return

    t_start = time.time()
    kept = 0
    with out_path.open("a") as fh:
        for i, (pid, title, desc, img_path) in enumerate(todo, 1):
            image_b64 = downscale_b64(img_path) if args.vision else None
            t0 = time.time()
            try:
                label, raw = call(args.model, title, desc, image_b64)
            except Exception as exc:
                print(f"  [{i}/{len(todo)}] FAILED {pid}: {exc}")
                continue
            dt = time.time() - t0

            record = {
                "photo_id": pid,
                "title": title,
                **label,
                "model": args.model,
                "backend": args.backend,
                "vision": args.vision,
                "codebook_version": "v6",
                "seconds": round(dt, 2),
                "prompt_tokens": raw.get("prompt_eval_count"),
                "output_tokens": raw.get("eval_count"),
                "cache_read_tokens": raw.get("cache_read"),
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()

            passed = label["gate_category"] == "astronomical_observation"
            kept += passed
            tag = label["subject"] if passed else f"({label['gate_category']})"
            eta = (time.time() - t_start) / i * (len(todo) - i) / 60
            print(f"  [{i}/{len(todo)}] {dt:5.1f}s  {tag:<20} {title[:44]:<44} eta {eta:.0f}m")

    elapsed = (time.time() - t_start) / 60
    print(f"\n{len(todo)} labelled in {elapsed:.1f}m — {kept} passed the gate")
    print(f"appended to data/{args.out}")


if __name__ == "__main__":
    main()
