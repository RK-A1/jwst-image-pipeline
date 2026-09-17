"""
Labelling with Claude (or a local model through Ollama), and the label log.

The label log, data/labels/raw_labels.jsonl, is the system of record for labels.
Model calls cost money, so every response is appended to it the moment it arrives
and the log is committed to git. The warehouse `labels` table is a projection of the
log, rebuilt from it by load_log(). A fresh clone can therefore rebuild the whole
dataset from Flickr without paying for labelling again.

Relabelling a photo appends a new record, and the latest record per photo_id wins.
"""

import base64
import html
import io
import json
import logging
import os
import re
import urllib.request
from pathlib import Path

from include.jwst_pipeline import codebook
from include.jwst_pipeline.config import LAUNCH_DATE

log = logging.getLogger(__name__)

DEFAULT_API_MODEL = "claude-haiku-4-5"   # the model the published dataset was built with
DEFAULT_OLLAMA_MODEL = "gemma4:12b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/api/chat")
DESC_CHARS = 4000        # p99 caption is 3,737 chars; this keeps the alt-text block
IMAGE_LONG_EDGE = 896    # downscale before sending; originals run past 100 megapixels

# Column order of the warehouse labels table.
LABEL_COLUMNS = [
    "photo_id", "gate_category", "subject", "modality", "object_name", "instrument",
    "confidence", "rationale", "model", "backend", "vision", "codebook_version",
    "seconds", "prompt_tokens", "output_tokens", "cache_read_tokens",
]

_ENUMS = {
    "gate_category": codebook.GATE_CATEGORIES,
    "subject": codebook.SUBJECTS,
    "modality": codebook.MODALITIES,
    "instrument": codebook.INSTRUMENTS,
    "confidence": codebook.CONFIDENCES,
}


class LabelLogError(RuntimeError):
    pass


# ── prompt inputs ───────────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")


def clean_caption(desc: str | None) -> str:
    """Strip HTML and decode entities. Flickr stores raw markup, and one
    `<a href="..." rel="noreferrer nofollow">` spends ~75 characters saying nothing."""
    return html.unescape(_TAG_RE.sub("", desc or "")).strip()


def user_text(title: str, desc: str | None) -> str:
    """
    Clean BEFORE truncating. 45% of these captions end with a NASA/ESA "Image
    description:" accessibility block describing the visible content, and it sits at
    the very end, where a short window cuts it off.
    """
    return f"TITLE: {title}\n\nCAPTION: {clean_caption(desc)[:DESC_CHARS]}"


def encode_image(path: Path) -> str | None:
    """Base64 JPEG downscaled to IMAGE_LONG_EDGE, or None if unreadable."""
    from PIL import Image

    from include.jwst_pipeline.images import open_rgb

    try:
        img = open_rgb(path)
    except Exception:
        return None
    img.thumbnail((IMAGE_LONG_EDGE, IMAGE_LONG_EDGE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ── model backends ──────────────────────────────────────────────────────────────

def call_api(client, model: str, title: str, desc: str | None,
             image_b64: str | None) -> tuple[dict, dict]:
    """
    Hosted inference. The schema is enforced by giving it to a single strict tool and
    forcing that tool, so the response is always a valid object rather than prose.
    """
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
        system=[{"type": "text", "text": codebook.SYSTEM, "cache_control": {"type": "ephemeral"}}],
        tools=[{
            "name": "label_photo",
            "description": "Record the classification for this photo.",
            "input_schema": codebook.SCHEMA,
            "strict": True,
        }],
        tool_choice={"type": "tool", "name": "label_photo"},
        messages=[{"role": "user", "content": content}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            usage = resp.usage
            return dict(block.input), {
                "prompt_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": getattr(usage, "cache_read_input_tokens", None),
            }
    raise ValueError(f"no tool_use block in response (stop_reason={resp.stop_reason})")


def call_ollama(model: str, title: str, desc: str | None,
                image_b64: str | None) -> tuple[dict, dict]:
    """Local inference. Ollama constrains decoding to the schema through `format`."""
    msg = {"role": "user", "content": user_text(title, desc)}
    if image_b64:
        msg["images"] = [image_b64]
    body = json.dumps({
        "model": model,
        "stream": False,
        "format": codebook.SCHEMA,
        "options": {"temperature": 0, "num_ctx": 4096},
        "messages": [{"role": "system", "content": codebook.SYSTEM}, msg],
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return json.loads(raw["message"]["content"]), {
        "prompt_tokens": raw.get("prompt_eval_count"),
        "output_tokens": raw.get("eval_count"),
        "cache_read_tokens": None,
    }


# ── the queue ───────────────────────────────────────────────────────────────────

def queue(con, log_path: Path, limit: int) -> list[tuple[str, str, str, str]]:
    """
    Photos still to label: downloaded, never labelled, and not dated before launch.
    Ids already in the log are excluded even if the warehouse has not loaded them yet,
    so two runs close together never pay for the same photo twice.
    """
    already = {r["photo_id"] for r in read_log(log_path)}
    rows = con.execute("""
        SELECT photo_id, title, description, image_file FROM photos
        WHERE image_file IS NOT NULL
          AND photo_id NOT IN (SELECT photo_id FROM labels)
          AND (date_taken IS NULL OR date_taken >= CAST(? AS TIMESTAMP))
        ORDER BY photo_id
    """, [LAUNCH_DATE]).fetchall()
    return [r for r in rows if r[0] not in already][:limit]


# ── the log ─────────────────────────────────────────────────────────────────────

def build_record(photo_id: str, title: str, label: dict, usage: dict, *, model: str,
                 backend: str, vision: bool, seconds: float) -> dict:
    return {
        "photo_id": photo_id,
        "title": title,
        **label,
        "model": model,
        "backend": backend,
        "vision": vision,
        "codebook_version": codebook.CODEBOOK_VERSION,
        "seconds": round(seconds, 2),
        **usage,
    }


def append_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_log(path: Path) -> list[dict]:
    """
    Every record in the log. A malformed final line is skipped with a warning, since
    it is most likely a write still in progress; malformed lines anywhere else mean
    the log is damaged and raise.
    """
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    records = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if n == len(lines):
                log.warning("skipping incomplete final line %d of %s", n, path.name)
                continue
            raise LabelLogError(f"{path.name} line {n} is not valid JSON") from None
    return records


def validate(record: dict) -> list[str]:
    """Problems with a record, or an empty list. This is the check that catches an
    out-of-enum gate value before it silently becomes a rejection."""
    problems = [
        f"{field}={record.get(field)!r}"
        for field, allowed in _ENUMS.items()
        if record.get(field) not in allowed
    ]
    if not record.get("photo_id"):
        problems.append("missing photo_id")
    if record.get("gate_category") == codebook.KEEP_GATE and codebook.NOT_APPLICABLE in (
        record.get("subject"), record.get("modality")
    ):
        problems.append("observation with subject or modality not_applicable")
    return problems


def load_log(con, path: Path) -> int:
    """Rebuild the labels table from the log, atomically. Returns the row count."""
    latest: dict[str, dict] = {}
    for record in read_log(path):
        latest[record["photo_id"]] = record

    invalid = {pid: p for pid, r in latest.items() if (p := validate(r))}
    if invalid:
        sample = "; ".join(f"{pid}: {', '.join(p)}" for pid, p in list(invalid.items())[:10])
        raise LabelLogError(
            f"{len(invalid)} label record(s) violate the codebook and would be "
            f"misread as rejections. Relabel them. {sample}"
        )

    rows = [tuple(r.get(c) for c in LABEL_COLUMNS) for r in latest.values()]
    con.begin()
    try:
        con.execute("DELETE FROM labels")
        if rows:
            placeholders = ", ".join("?" * len(LABEL_COLUMNS))
            con.executemany(
                f"INSERT INTO labels ({', '.join(LABEL_COLUMNS)}) VALUES ({placeholders})", rows
            )
        con.commit()
    except Exception:
        con.rollback()
        raise
    return len(rows)
