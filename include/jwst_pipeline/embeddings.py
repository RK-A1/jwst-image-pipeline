"""
ResNet50 image embeddings.

The vectors drive near-duplicate grouping in the dataset build. The model, weights
and preprocessing are unchanged from the original pipeline, so the 4,335 vectors
carried over from it remain comparable with any computed here.

Computing and storing are separate steps. Batches are embedded in parallel into
staging files, and one task then loads them all, because DuckDB accepts a single
writing process.
"""

import logging
import os
from pathlib import Path

import numpy as np

from include.jwst_pipeline import images
from include.jwst_pipeline.config import paths

log = logging.getLogger(__name__)

MODEL_NAME = "resnet50-imagenet1k-v2"
DIM = 2048
BATCH_SIZE = 32
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def device():
    """
    Best available torch device. Docker Desktop on a Mac gives Linux containers no GPU
    access, so under Astro this is the CPU; MPS applies when run directly on the host.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(dev):
    import torch.nn as nn
    import torchvision.models as models

    # Keep downloaded weights on the mounted volume so a new container does not refetch.
    os.environ.setdefault("TORCH_HOME", str(paths().torch_cache))
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()  # drop the classifier head: output is the 2048-dim pool
    return model.to(dev).eval()


def _transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def embed(items: list[tuple[str, Path]], model, dev) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """
    Embed (photo_id, path) pairs. Returns the ids embedded, their vectors in the same
    order, and an error message for each image that could not be read.
    """
    import torch

    transform = _transform()
    tensors, ids, errors = [], [], {}
    for photo_id, path in items:
        try:
            tensors.append(transform(images.open_rgb(path)))
            ids.append(photo_id)
        except Exception as exc:
            errors[photo_id] = f"{type(exc).__name__}: {exc}"
            log.warning("cannot embed %s: %s", photo_id, errors[photo_id])

    if not tensors:
        return [], np.empty((0, DIM), dtype=np.float32), errors
    with torch.no_grad():
        vectors = model(torch.stack(tensors).to(dev)).cpu().numpy().astype(np.float32)
    return ids, vectors, errors


def save_batch(path: Path, ids: list[str], vectors: np.ndarray, errors: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("wb") as fh:
        np.savez(
            fh,
            ids=np.array(ids, dtype=str),
            vectors=vectors.astype(np.float32).reshape(-1, DIM),
            error_ids=np.array(list(errors), dtype=str),
            error_messages=np.array(list(errors.values()), dtype=str),
        )
    os.replace(tmp, path)


def load_batch(path: Path) -> tuple[list[str], np.ndarray, dict[str, str]]:
    with np.load(path) as npz:
        errors = dict(zip(npz["error_ids"].tolist(), npz["error_messages"].tolist()))
        return npz["ids"].tolist(), npz["vectors"], errors


def pending(con, limit: int | None = None) -> list[tuple[str, str]]:
    """Downloaded photos with neither an embedding nor a recorded failure."""
    sql = """
        SELECT photo_id, image_file FROM photos
        WHERE image_file IS NOT NULL
          AND photo_id NOT IN (SELECT photo_id FROM embeddings)
          AND photo_id NOT IN (SELECT photo_id FROM embedding_errors)
        ORDER BY photo_id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    return con.execute(sql).fetchall()


def store(con, batch_files: list[Path], *, now) -> tuple[int, int]:
    """Load staged batches in one transaction. Returns (vectors stored, errors recorded)."""
    stored = failed = 0
    con.begin()
    try:
        for path in batch_files:
            ids, vectors, errors = load_batch(path)
            if ids:
                con.executemany(
                    "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                    [(pid, MODEL_NAME, vec.tolist(), now) for pid, vec in zip(ids, vectors)],
                )
                stored += len(ids)
            if errors:
                con.executemany(
                    "INSERT OR REPLACE INTO embedding_errors VALUES (?, ?, ?)",
                    [(pid, msg, now) for pid, msg in errors.items()],
                )
                failed += len(errors)
        con.commit()
    except Exception:
        con.rollback()
        raise
    return stored, failed
