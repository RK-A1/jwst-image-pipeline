"""
jwst_flickr_ingest — Daily incremental ingestion of JWST photos from Flickr.

Pipeline:
  ensure_schema
      │
  get_existing_ids       ← all photo_ids already in DuckDB
      │
  fetch_photos_metadata  ← paginate flickr.people.getPublicPhotos for ALL photos,
      │                    diff against existing IDs, then flickr.photos.getInfo
      │                    for each new photo's tags/description
  download_images        ← fetch largest available size → include/images/{id}.jpg
      │
  insert_records         ← upsert into DuckDB photos table
      │
  predict_labels         ← load best model from training_runs, run inference
                           on photos where embedding IS NOT NULL and
                           predicted_label IS NULL; skip if no model exists yet
"""

import logging
import os
import time
from pathlib import Path

import requests
from airflow.sdk import dag, task
from pendulum import datetime as pendulum_datetime

log = logging.getLogger(__name__)

FLICKR_USER = "nasawebbtelescope"
IMAGES_DIR = Path(__file__).resolve().parents[1] / "include" / "images"
PER_PAGE = 500
# Preferred download sizes, largest first
_SIZE_PREFERENCE = [
    "Original",
    "Large 2048",
    "Large 1600",
    "Large",
    "Medium 800",
    "Medium 640",
    "Medium",
]


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_INFER_BATCH        = 32
_CONFIDENCE_THRESH  = 0.6   # predictions below this confidence become 'unclassified'


def _flickr_client():
    """Return an unauthenticated flickrapi client (JSON format)."""
    import flickrapi

    api_key = os.environ["FLICKR_API_KEY"]
    return flickrapi.FlickrAPI(api_key, "", format="parsed-json")


def _get_device():
    import torch
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _infer_xgboost(
    model_path: str,
    rows: list,          # [(photo_id, embedding, image_path), ...]
    int_to_label: dict,
) -> dict[str, str]:
    """Run XGBoost inference on stored embedding vectors."""
    import numpy as np
    import xgboost as xgb

    photo_ids  = [r[0] for r in rows]
    embeddings = np.array([r[1] for r in rows], dtype=np.float32)

    clf = xgb.XGBClassifier()
    clf.load_model(model_path)

    proba = clf.predict_proba(embeddings)   # (N, num_classes)
    results = {}
    for pid, prob_row in zip(photo_ids, proba):
        confidence = float(prob_row.max())
        if confidence >= _CONFIDENCE_THRESH:
            results[pid] = int_to_label.get(int(prob_row.argmax()), "unknown")
        else:
            results[pid] = "unclassified"
    return results


def _infer_resnet(
    model_path: str,
    rows: list,          # [(photo_id, embedding, image_path), ...]
) -> dict[str, str]:
    """
    Run fine-tuned ResNet50 inference on raw images.

    Photos whose image_path is missing or unreadable are skipped (they keep
    predicted_label=NULL and will be retried on the next DAG run).
    label_to_int is loaded from the checkpoint so it always matches training.
    """
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image, UnidentifiedImageError

    Image.MAX_IMAGE_PIXELS = None  # JWST images can exceed PIL's default limit

    checkpoint   = torch.load(model_path, map_location="cpu", weights_only=False)
    label_to_int = checkpoint["label_to_int"]
    num_classes  = checkpoint["num_classes"]
    int_to_label = {v: k for k, v in label_to_int.items()}

    device = _get_device()
    model  = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model  = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    # Only rows with an image file on disk are usable
    valid = [(r[0], r[2]) for r in rows if r[2] and Path(r[2]).exists()]
    skipped = len(rows) - len(valid)
    if skipped:
        log.warning(
            "%d photo(s) skipped for ResNet inference — image file not found", skipped
        )

    predictions: dict[str, str] = {}
    for i in range(0, len(valid), _INFER_BATCH):
        batch       = valid[i : i + _INFER_BATCH]
        tensors, ids = [], []
        for photo_id, image_path in batch:
            try:
                img = Image.open(image_path).convert("RGB")
                tensors.append(transform(img))
                ids.append(photo_id)
            except (UnidentifiedImageError, OSError) as exc:
                log.warning("Cannot open %s for inference: %s", photo_id, exc)

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            proba  = torch.softmax(logits, dim=1).cpu()

        for photo_id, prob_row in zip(ids, proba):
            confidence = float(prob_row.max())
            if confidence >= _CONFIDENCE_THRESH:
                predictions[photo_id] = int_to_label.get(int(prob_row.argmax()), "unknown")
            else:
                predictions[photo_id] = "unclassified"

    return predictions


@dag(
    dag_id="jwst_flickr_ingest",
    start_date=pendulum_datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={"owner": "jwst", "retries": 2, "retry_delay": 30},
    tags=["jwst", "flickr", "ingest"],
    doc_md=__doc__,
)
def jwst_flickr_ingest():

    @task
    def ensure_schema() -> None:
        """Create DuckDB tables if they don't exist yet."""
        from include.db import DB_PATH, init_schema

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        init_schema()
        log.info("Schema ready at %s", DB_PATH)

    @task
    def get_existing_ids() -> list[str]:
        """Return all photo_ids already in DuckDB."""
        from include.db import get_conn

        con = get_conn(read_only=True)
        rows = con.execute("SELECT photo_id FROM photos").fetchall()
        con.close()
        ids = [r[0] for r in rows]
        log.info("Existing photos in DB: %d", len(ids))
        return ids

    @task
    def fetch_photos_metadata(existing_ids: list[str]) -> list[dict]:
        """
        Paginate flickr.people.getPublicPhotos for ALL photos, filter out
        IDs already in DuckDB, then call flickr.photos.getInfo for each
        new photo to get the full tag list, description, and taken date.

        Returns a list of dicts ready for download_images / insert_records.
        """
        flickr = _flickr_client()

        # Resolve path alias → NSID (cheap, cached by Flickr)
        user_resp = flickr.urls.lookupUser(url=f"https://www.flickr.com/photos/{FLICKR_USER}/")
        user_id = user_resp["user"]["id"]
        log.info("Resolved '%s' → NSID %s", FLICKR_USER, user_id)

        # ── 1. Collect ALL photo IDs via paginated getPublicPhotos ──────────
        all_ids: list[str] = []
        page = 1
        while True:
            resp = flickr.people.getPublicPhotos(
                user_id=user_id, per_page=PER_PAGE, page=page,
            )
            page_data = resp["photos"]
            batch = page_data["photo"]
            total_pages = page_data["pages"]

            all_ids.extend(p["id"] for p in batch)
            log.info(
                "getPublicPhotos page %d/%d: %d photos (running total %d)",
                page,
                total_pages,
                len(batch),
                len(all_ids),
            )

            if page >= total_pages:
                break
            page += 1
            time.sleep(0.3)  # ~3 req/s, well under the 3600 req/hr cap

        # ── 2. Filter to only new photos ────────────────────────────────────
        existing_set = set(existing_ids)
        new_ids = [pid for pid in all_ids if pid not in existing_set]
        log.info(
            "Flickr total: %d | Already in DB: %d | New: %d",
            len(all_ids), len(existing_set), len(new_ids),
        )

        if not new_ids:
            log.info("No new photos to ingest.")
            return []

        # ── 3. Call getInfo for each new photo to get tags + metadata ───────
        results: list[dict] = []
        for photo_id in new_ids:
            try:
                info = flickr.photos.getInfo(photo_id=photo_id)["photo"]

                tags = [
                    t["_content"]
                    for t in info.get("tags", {}).get("tag", [])
                ]
                results.append(
                    {
                        "photo_id": photo_id,
                        "title": info["title"]["_content"],
                        "description": info["description"]["_content"],
                        "tags": tags,
                        # Flickr format: "YYYY-MM-DD HH:MM:SS"
                        "date_taken": info["dates"]["taken"],
                    }
                )
            except Exception as exc:
                log.warning("getInfo failed for photo %s: %s", photo_id, exc)

            time.sleep(0.3)

        log.info("Successfully fetched metadata for %d new photos", len(results))
        return results

    @task
    def download_images(photos: list[dict]) -> list[dict]:
        """
        For each photo, download the largest available Flickr size to
        include/images/{photo_id}.jpg.  Already-present files are skipped.
        Returns the input list enriched with an 'image_path' key.
        """
        if not photos:
            log.info("No photos to download.")
            return []

        flickr = _flickr_client()
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        enriched: list[dict] = []
        for photo in photos:
            photo_id = photo["photo_id"]
            dest = IMAGES_DIR / f"{photo_id}.jpg"

            if dest.exists():
                log.info("Skip (already downloaded): %s", photo_id)
                enriched.append({**photo, "image_path": str(dest)})
                continue

            try:
                sizes_resp = flickr.photos.getSizes(photo_id=photo_id)
                sizes = sizes_resp["sizes"]["size"]

                # Pick largest preferred size, fall back to last entry
                url: str | None = None
                for label in _SIZE_PREFERENCE:
                    match = next((s for s in sizes if s["label"] == label), None)
                    if match:
                        url = match["source"]
                        break
                if url is None:
                    url = sizes[-1]["source"]

                r = requests.get(url, timeout=60, stream=True)
                r.raise_for_status()
                with dest.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=65_536):
                        fh.write(chunk)

                log.info("Downloaded %s → %s", photo_id, dest.name)
                enriched.append({**photo, "image_path": str(dest)})

            except Exception as exc:
                log.warning("Download failed for %s: %s", photo_id, exc)
                # Still pass the record through; insert_records will skip it
                enriched.append({**photo, "image_path": None})

            time.sleep(0.3)

        downloaded = sum(1 for p in enriched if p["image_path"])
        log.info("Downloaded %d/%d images", downloaded, len(enriched))
        return enriched

    @task
    def insert_records(photos: list[dict]) -> int:
        """
        Upsert photo metadata into DuckDB.  Records whose image_path is None
        (download failed) are skipped.  Existing rows are updated in place so
        the task is safely re-runnable.
        """
        from datetime import datetime, timezone

        from include.db import get_conn

        if not photos:
            log.info("No records to insert.")
            return 0

        con = get_conn()
        now = datetime.now(timezone.utc)
        inserted = 0

        for p in photos:
            if p.get("image_path") is None:
                log.debug("Skipping %s — no image downloaded", p["photo_id"])
                continue
            try:
                con.execute(
                    """
                    INSERT INTO photos (
                        photo_id, title, description, tags,
                        image_path, date_taken, date_ingested
                    )
                    VALUES (?, ?, ?, ?, ?, ?::TIMESTAMP, ?)
                    ON CONFLICT (photo_id) DO UPDATE SET
                        title        = excluded.title,
                        description  = excluded.description,
                        tags         = excluded.tags,
                        image_path   = excluded.image_path,
                        date_taken   = excluded.date_taken
                    """,
                    [
                        p["photo_id"],
                        p["title"],
                        p["description"],
                        p["tags"],
                        p["image_path"],
                        p["date_taken"],
                        now,
                    ],
                )
                inserted += 1
            except Exception as exc:
                log.warning("Insert failed for %s: %s", p["photo_id"], exc)

        con.close()
        log.info("Upserted %d records into photos table", inserted)
        return inserted

    @task
    def predict_labels(inserted_count: int) -> int:
        """
        Find photos that have an embedding but no predicted_label, load the
        best model from training_runs (ranked by f1_score), run inference, and
        write predicted_label back to DuckDB.

        Skips gracefully when:
          - the training_runs table doesn't exist yet
          - no trained models are recorded
          - the model file is missing from disk
          - no photos need labeling

        For XGBoost models the label mapping is reconstructed from the
        canonical_labels present in the photos table (same sort order used
        during training).  For ResNet checkpoints the mapping is loaded
        directly from the saved dict.

        Returns the number of photos labeled.
        """
        from include.db import get_conn

        con = get_conn(read_only=True)

        # ── 1. Find best model ─────────────────────────────────────────────
        try:
            best = con.execute(
                """
                SELECT run_id, model_type, f1_score, model_path
                FROM   training_runs
                WHERE  model_path IS NOT NULL
                ORDER  BY f1_score DESC NULLS LAST
                LIMIT  1
                """
            ).fetchone()
        except Exception as exc:
            # training_runs doesn't exist yet
            log.info("training_runs not available yet, skipping inference: %s", exc)
            con.close()
            return 0

        if best is None:
            log.info("No trained models in training_runs yet, skipping inference.")
            con.close()
            return 0

        run_id, model_type, f1, model_path = best
        log.info("Best model: %s  type=%s  f1=%.4f  path=%s", run_id, model_type, f1, model_path)

        if not Path(model_path).exists():
            log.warning("Model file not found on disk: %s — skipping inference.", model_path)
            con.close()
            return 0

        # ── 2. Fetch photos needing labels ─────────────────────────────────
        rows = con.execute(
            """
            SELECT photo_id, embedding, image_path
            FROM   photos
            WHERE  predicted_label IS NULL
              AND  embedding IS NOT NULL
            """
        ).fetchall()

        if not rows:
            log.info("All photos already have a predicted_label.")
            con.close()
            return 0

        log.info("%d photo(s) need a predicted_label (model_type=%s)", len(rows), model_type)

        # ── 3. Build int → label mapping ───────────────────────────────────
        # XGBoost checkpoints don't store metadata, so we reconstruct the
        # same sorted(distinct canonical_label) mapping used at training time.
        # ResNet checkpoints include label_to_int directly.
        if model_type == "xgboost":
            try:
                label_rows = con.execute(
                    """
                    SELECT canonical_label
                    FROM (
                        SELECT canonical_label, count(*) AS n
                        FROM   photos
                        WHERE  canonical_label IS NOT NULL
                          AND  canonical_label NOT IN ('unclassified', 'observatory / engineering')
                        GROUP  BY canonical_label
                        HAVING count(*) >= 5
                    )
                    ORDER BY canonical_label
                    """
                ).fetchall()
            except Exception:
                log.warning("canonical_label column not found — skipping predict_labels")
                con.close()
                return 0
            int_to_label = {i: r[0] for i, r in enumerate(label_rows)}
        else:
            int_to_label = {}   # loaded from checkpoint inside _infer_resnet

        con.close()

        # ── 4. Run inference ───────────────────────────────────────────────
        if model_type == "xgboost":
            predictions = _infer_xgboost(model_path, rows, int_to_label)
        elif model_type == "resnet_finetune":
            predictions = _infer_resnet(model_path, rows)
        else:
            log.warning("Unknown model_type '%s', skipping inference.", model_type)
            return 0

        if not predictions:
            log.warning("Inference produced no predictions.")
            return 0

        # ── 5. Write predicted_label back ──────────────────────────────────
        con = get_conn()
        updated = 0
        for photo_id, label in predictions.items():
            try:
                con.execute(
                    "UPDATE photos SET predicted_label = ? WHERE photo_id = ?",
                    [label, photo_id],
                )
                updated += 1
            except Exception as exc:
                log.warning("Failed to write label for %s: %s", photo_id, exc)
        con.close()

        log.info("Wrote predicted_label for %d/%d photo(s).", updated, len(rows))
        return updated

    # ── Wire up the DAG ───────────────────────────────────────────────────────
    schema      = ensure_schema()
    existing    = get_existing_ids()
    metadata    = fetch_photos_metadata(existing)
    downloaded  = download_images(metadata)
    inserted    = insert_records(downloaded)
    predict_labels(inserted)

    # ensure_schema must complete before we query the table for existing IDs
    schema >> existing


jwst_flickr_ingest()
