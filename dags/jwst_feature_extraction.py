"""
jwst_feature_extraction — Extract ResNet50 embeddings for unprocessed photos.

Pipeline:
  get_unembedded_ids
      │
  extract_and_store_embeddings   ← batched, MPS-accelerated
      (one mapped task instance per batch of 32)

ResNet50 is used with its classification head replaced by nn.Identity(), giving
a 2048-dim feature vector per image.  The model weights are loaded once per
task instance (i.e. once per batch) and never written to disk.

ImageNet normalisation is applied so the pretrained weights are used correctly:
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]
"""

import logging
from pathlib import Path

from airflow.sdk import dag, task
from pendulum import datetime as pendulum_datetime

log = logging.getLogger(__name__)

BATCH_SIZE = 32
# Matches what ResNet was trained on
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_device():
    """Return the best available torch device: MPS → CPU."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(device):
    """
    Return ResNet50 with the FC head replaced by Identity, in eval mode,
    moved to device.  Uses IMAGENET1K_V2 weights (best available).
    """
    import os
    import torch.nn as nn
    import torchvision.models as models

    # Cache weights in include/ so they survive container restarts
    os.environ.setdefault(
        "TORCH_HOME",
        str(Path(__file__).resolve().parents[1] / "include" / ".torch_cache"),
    )
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()  # strip classifier → output is 2048-dim
    model = model.to(device)
    model.eval()
    return model


def _build_transform():
    """Standard ImageNet pre-processing pipeline for ResNet."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


@dag(
    dag_id="jwst_feature_extraction",
    start_date=pendulum_datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={"owner": "jwst", "retries": 1, "retry_delay": 60},
    tags=["jwst", "embeddings", "ml"],
    doc_md=__doc__,
)
def jwst_feature_extraction():

    @task
    def get_unembedded_ids() -> list[list[str]]:
        """
        Query DuckDB for photos whose embedding is NULL and whose image file
        exists on disk.  Returns photo_ids split into batches of BATCH_SIZE
        ready for dynamic task mapping.
        """
        from include.db import get_conn

        con = get_conn(read_only=True)
        rows = con.execute(
            "SELECT photo_id, image_path FROM photos WHERE embedding IS NULL"
        ).fetchall()
        con.close()

        # Filter to rows where the file is actually present
        valid = [
            row[0]
            for row in rows
            if row[1] and Path(row[1]).exists()
        ]

        missing_files = len(rows) - len(valid)
        if missing_files:
            log.warning(
                "%d photo(s) skipped — image_path missing or file not found",
                missing_files,
            )

        if not valid:
            log.info("No photos need embedding.")
            return []

        # Split into batches for dynamic task mapping
        batches = [
            valid[i : i + BATCH_SIZE]
            for i in range(0, len(valid), BATCH_SIZE)
        ]
        log.info(
            "%d photos → %d batch(es) of up to %d",
            len(valid),
            len(batches),
            BATCH_SIZE,
        )
        return batches

    @task(max_active_tis_per_dagrun=2)
    def extract_and_store_embeddings(photo_id_batch: list[str]) -> int:
        """
        For a single batch:
          1. Load images from disk and apply ImageNet transforms.
          2. Run a forward pass through ResNet50 (MPS-accelerated).
          3. Write the 2048-dim float vectors back to DuckDB.

        Returns the number of embeddings successfully stored.
        """
        import torch
        from PIL import Image, UnidentifiedImageError

        Image.MAX_IMAGE_PIXELS = None  # JWST images can exceed PIL's default bomb limit

        from include.db import get_conn

        if not photo_id_batch:
            return 0

        device = _get_device()
        log.info("Using device: %s", device)

        model = _build_model(device)
        transform = _build_transform()

        # ── Load image paths from DB ──────────────────────────────────────────
        con = get_conn(read_only=True)
        placeholders = ", ".join("?" * len(photo_id_batch))
        path_rows = con.execute(
            f"SELECT photo_id, image_path FROM photos WHERE photo_id IN ({placeholders})",
            photo_id_batch,
        ).fetchall()
        con.close()

        id_to_path = {row[0]: row[1] for row in path_rows}

        # ── Preprocess images ─────────────────────────────────────────────────
        tensors: list[torch.Tensor] = []
        valid_ids: list[str] = []

        for photo_id in photo_id_batch:
            image_path = id_to_path.get(photo_id)
            if not image_path or not Path(image_path).exists():
                log.warning("Image not found for %s, skipping", photo_id)
                continue
            try:
                img = Image.open(image_path).convert("RGB")
                tensors.append(transform(img))
                valid_ids.append(photo_id)
            except (UnidentifiedImageError, OSError) as exc:
                log.warning("Could not open %s (%s): %s", photo_id, image_path, exc)

        if not tensors:
            log.warning("No valid images in batch, nothing to embed.")
            return 0

        # ── Forward pass ──────────────────────────────────────────────────────
        batch_tensor = torch.stack(tensors).to(device)  # (N, 3, 224, 224)

        with torch.no_grad():
            embeddings = model(batch_tensor)             # (N, 2048)

        # Move back to CPU and convert to plain Python lists for DuckDB storage
        embeddings_cpu = embeddings.cpu().float().tolist()  # list[list[float]]

        # ── Write embeddings to DuckDB ────────────────────────────────────────
        # Open a fresh read-write connection only for the write phase.
        con = get_conn()
        stored = 0
        for photo_id, embedding in zip(valid_ids, embeddings_cpu):
            try:
                con.execute(
                    "UPDATE photos SET embedding = ? WHERE photo_id = ?",
                    [embedding, photo_id],
                )
                stored += 1
            except Exception as exc:
                log.warning("Failed to store embedding for %s: %s", photo_id, exc)

        con.close()
        log.info("Stored %d/%d embeddings (device: %s)", stored, len(valid_ids), device)
        return stored

    # ── Wire up ───────────────────────────────────────────────────────────────
    batches = get_unembedded_ids()
    extract_and_store_embeddings.expand(photo_id_batch=batches)


jwst_feature_extraction()
