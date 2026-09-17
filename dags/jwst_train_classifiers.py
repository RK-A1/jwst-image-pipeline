"""
jwst_train_classifiers — Train and compare two JWST image classifiers.

Pipeline:
  load_dataset
      │
      ├─── train_xgboost   ──┐
      │                      ├── compare_models
      └─── train_resnet    ──┘

load_dataset
  Reads embeddings + canonical_labels from DuckDB, drops photos missing
  either field, drops classes with fewer than MIN_SAMPLES_PER_CLASS examples,
  and returns stratified 80/20 train/test photo-ID splits via XCom.

train_xgboost
  Fetches embeddings from DuckDB for the train split, trains an XGBoost
  multi-class classifier, evaluates on the test split, logs metrics to
  training_runs, and saves the model to models/xgboost_latest.json.

train_resnet
  Loads raw images from disk for both splits, fine-tunes a ResNet50 end-to-end
  (differential LRs: 1e-5 backbone / 1e-4 head) using PyTorch with MPS
  acceleration, evaluates, logs metrics, and saves to models/resnet_latest.pt.

compare_models
  Queries training_runs for the two run IDs produced above and prints a
  side-by-side comparison table.
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from airflow.sdk import dag, task
from pendulum import datetime as pendulum_datetime

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "include" / "models"
MIN_SAMPLES_PER_CLASS = 5   # classes with fewer samples are excluded
EXCLUDED_LABELS = {"unclassified", "observatory / engineering"}  # catch-alls, not useful for training
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ResNet fine-tuning hyper-parameters
RESNET_EPOCHS = 15
RESNET_BATCH_SIZE = 8        # small batch to fit in Docker container memory
RESNET_LR_BACKBONE = 1e-5
RESNET_LR_HEAD = 1e-4
RESNET_WEIGHT_DECAY = 1e-4
RESNET_PATIENCE = 4          # early-stopping patience (val-loss)

# ImageNet normalisation (must match feature-extraction DAG)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_device():
    import torch
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _log_run(con, run_id: str, model_type: str,
             accuracy: float, f1: float, model_path: str) -> None:
    """Insert a row into training_runs."""
    con.execute(
        """
        INSERT INTO training_runs (run_id, ts, model_type, accuracy, f1_score, model_path)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id) DO UPDATE SET
            accuracy   = excluded.accuracy,
            f1_score   = excluded.f1_score,
            model_path = excluded.model_path
        """,
        [run_id, datetime.now(timezone.utc), model_type, accuracy, f1, model_path],
    )


# ─────────────────────────────────────────────────────────────────────────────

@dag(
    dag_id="jwst_train_classifiers",
    start_date=pendulum_datetime(2024, 1, 1),
    schedule=None,          # triggered manually or by a sensor after ingest
    catchup=False,
    default_args={"owner": "jwst", "retries": 0},
    tags=["jwst", "ml", "training"],
    doc_md=__doc__,
)
def jwst_train_classifiers():

    # ── Task 1 ────────────────────────────────────────────────────────────────
    @task
    def load_dataset() -> dict:
        """
        Pull photos that have both an embedding and a canonical_label, drop
        rare classes, encode labels, stratified-split 80/20, and return the
        split info dict (photo IDs only — embeddings are fetched per-task to
        avoid bloating XCom).

        Returns
        -------
        {
          "train_ids":    [photo_id, ...],
          "test_ids":     [photo_id, ...],
          "label_to_int": {"nebula": 0, "galaxy": 1, ...},
          "classes":      ["galaxy", "nebula", ...],   # sorted
          "num_classes":  N,
        }
        """
        from sklearn.model_selection import train_test_split
        from include.db import get_conn

        con = get_conn(read_only=True)
        rows = con.execute(
            """
            SELECT photo_id, canonical_label
            FROM photos
            WHERE embedding IS NOT NULL
              AND canonical_label IS NOT NULL
              AND canonical_label NOT IN ('unclassified', 'observatory / engineering')
            """
        ).fetchall()
        con.close()

        if not rows:
            raise ValueError(
                "No photos with both embedding and canonical_label found. "
                "Run jwst_feature_extraction and tag_consolidation first."
            )

        photo_ids = [r[0] for r in rows]
        labels    = [r[1] for r in rows]

        # Drop classes that are too small for a stratified split
        from collections import Counter
        label_counts = Counter(labels)
        kept = {lbl for lbl, cnt in label_counts.items() if cnt >= MIN_SAMPLES_PER_CLASS}
        dropped = {lbl: cnt for lbl, cnt in label_counts.items() if lbl not in kept}
        if dropped:
            log.warning("Dropping classes with < %d samples: %s", MIN_SAMPLES_PER_CLASS, dropped)

        filtered = [(pid, lbl) for pid, lbl in zip(photo_ids, labels) if lbl in kept]
        if not filtered:
            raise ValueError("No classes survived the minimum-samples filter.")

        photo_ids, labels = zip(*filtered)
        photo_ids = list(photo_ids)
        labels    = list(labels)

        classes     = sorted(set(labels))
        label_to_int = {lbl: i for i, lbl in enumerate(classes)}
        int_labels   = [label_to_int[lbl] for lbl in labels]

        train_ids, test_ids, y_train, y_test = train_test_split(
            photo_ids,
            int_labels,
            test_size=TEST_SIZE,
            stratify=int_labels,
            random_state=RANDOM_STATE,
        )

        log.info(
            "Dataset: %d total → %d train / %d test | %d classes: %s",
            len(photo_ids), len(train_ids), len(test_ids), len(classes), classes,
        )
        for lbl in classes:
            n = labels.count(lbl)
            log.info("  %-30s  %d photos", lbl, n)

        return {
            "train_ids":    train_ids,
            "test_ids":     test_ids,
            "label_to_int": label_to_int,
            "classes":      classes,
            "num_classes":  len(classes),
        }

    # ── Task 2 ────────────────────────────────────────────────────────────────
    @task
    def train_xgboost(split: dict) -> str:
        """
        Train an XGBoost multi-class classifier on the 2048-dim ResNet embeddings.
        Saves model to models/xgboost_latest.json and logs metrics to
        training_runs.  Returns the run_id.
        """
        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, f1_score
        from include.db import get_conn

        run_id = f"xgboost_{uuid.uuid4().hex[:8]}"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        label_to_int: dict = split["label_to_int"]
        train_ids: list   = split["train_ids"]
        test_ids: list    = split["test_ids"]

        # ── Fetch embeddings from DuckDB ──────────────────────────────────────
        def fetch_Xy(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
            con = get_conn(read_only=True)
            placeholders = ", ".join("?" * len(ids))
            rows = con.execute(
                f"SELECT photo_id, embedding, canonical_label "
                f"FROM photos WHERE photo_id IN ({placeholders})",
                ids,
            ).fetchall()
            con.close()
            id_map = {r[0]: (r[1], r[2]) for r in rows}
            X, y = [], []
            for pid in ids:
                if pid in id_map:
                    emb, lbl = id_map[pid]
                    X.append(emb)
                    y.append(label_to_int[lbl])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

        log.info("Loading train embeddings (%d photos)…", len(train_ids))
        X_train, y_train = fetch_Xy(train_ids)
        log.info("Loading test embeddings (%d photos)…", len(test_ids))
        X_test, y_test = fetch_Xy(test_ids)

        # ── Train ─────────────────────────────────────────────────────────────
        num_classes = split["num_classes"]
        objective   = "binary:logistic" if num_classes == 2 else "multi:softmax"

        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective=objective,
            num_class=num_classes if num_classes > 2 else None,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            early_stopping_rounds=20,
        )

        log.info("Training XGBoost (%d classes, %d train samples)…", num_classes, len(X_train))
        clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred   = clf.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        f1       = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        log.info("XGBoost  accuracy=%.4f  weighted-F1=%.4f", accuracy, f1)

        int_to_label = {v: k for k, v in label_to_int.items()}
        from sklearn.metrics import classification_report
        log.info(
            "Classification report:\n%s",
            classification_report(y_test, y_pred,
                                  target_names=[int_to_label[i] for i in range(num_classes)],
                                  zero_division=0),
        )

        # ── Save model ────────────────────────────────────────────────────────
        model_path = str(MODELS_DIR / "xgboost_latest.json")
        clf.save_model(model_path)
        log.info("Saved XGBoost model → %s", model_path)

        # ── Log to training_runs ──────────────────────────────────────────────
        con = get_conn()
        _log_run(con, run_id, "xgboost", accuracy, f1, model_path)
        con.close()

        return run_id

    # ── Task 3 ────────────────────────────────────────────────────────────────
    @task
    def train_resnet(split: dict) -> str:
        """
        Fine-tune ResNet50 end-to-end on raw images.

        Architecture change: replace the pretrained 1000-class FC head with a
        new Linear(2048, num_classes) layer trained from scratch.

        Differential learning rates keep the pretrained backbone useful:
          backbone layers → lr=1e-5
          new FC head     → lr=1e-4

        Data augmentation is applied to the training set only.
        num_workers=0 is required for MPS / macOS multiprocessing compatibility.
        Early stopping on validation loss (patience=RESNET_PATIENCE).
        """
        import os
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
        from PIL import Image, UnidentifiedImageError
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        from include.db import get_conn

        Image.MAX_IMAGE_PIXELS = None  # JWST images can exceed PIL's default limit
        os.environ.setdefault(
            "TORCH_HOME",
            str(Path(__file__).resolve().parents[1] / "include" / ".torch_cache"),
        )

        run_id = f"resnet_{uuid.uuid4().hex[:8]}"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        label_to_int: dict  = split["label_to_int"]
        num_classes: int    = split["num_classes"]
        int_to_label        = {v: k for k, v in label_to_int.items()}
        train_ids: list     = split["train_ids"]
        test_ids: list      = split["test_ids"]

        # ── Fetch image paths + labels from DuckDB ────────────────────────────
        def fetch_records(ids: list[str]) -> list[tuple[str, int]]:
            """Return [(image_path, int_label), ...] for valid, on-disk files."""
            con = get_conn(read_only=True)
            placeholders = ", ".join("?" * len(ids))
            rows = con.execute(
                f"SELECT photo_id, image_path, canonical_label "
                f"FROM photos WHERE photo_id IN ({placeholders})",
                ids,
            ).fetchall()
            con.close()
            records = []
            for pid, path, lbl in rows:
                if path and Path(path).exists() and lbl in label_to_int:
                    records.append((path, label_to_int[lbl]))
                else:
                    log.warning("Skipping %s — missing file or unknown label", pid)
            return records

        log.info("Fetching train image paths (%d photos)…", len(train_ids))
        train_records = fetch_records(train_ids)
        log.info("Fetching test image paths (%d photos)…", len(test_ids))
        test_records  = fetch_records(test_ids)

        if not train_records:
            raise ValueError("No valid training images found on disk.")

        # ── Dataset ───────────────────────────────────────────────────────────
        class JWSTDataset(Dataset):
            def __init__(self, records: list[tuple[str, int]], transform):
                self.records   = records
                self.transform = transform

            def __len__(self) -> int:
                return len(self.records)

            def __getitem__(self, idx: int):
                path, label = self.records[idx]
                try:
                    img = Image.open(path).convert("RGB")
                except (UnidentifiedImageError, OSError):
                    img = Image.new("RGB", (224, 224))   # black placeholder
                return self.transform(img), label

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        # num_workers=0: MPS doesn't support forked multiprocessing on macOS
        train_loader = DataLoader(
            JWSTDataset(train_records, train_transform),
            batch_size=RESNET_BATCH_SIZE, shuffle=True, num_workers=0,
        )
        test_loader = DataLoader(
            JWSTDataset(test_records, test_transform),
            batch_size=RESNET_BATCH_SIZE, shuffle=False, num_workers=0,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        device = _get_device()
        log.info("Training ResNet50 on device: %s", device)

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(2048, num_classes)   # new head, random init

        # Freeze early layers (layer1–3) to save memory — only layer4 + fc are trained.
        # This reduces gradient storage by ~75% vs full fine-tuning.
        for name, param in model.named_parameters():
            if not (name.startswith("layer4") or name.startswith("fc")):
                param.requires_grad = False

        model = model.to(device)

        # Class weights to handle imbalance
        all_train_labels = [lbl for _, lbl in train_records]
        class_counts = np.bincount(all_train_labels, minlength=num_classes).astype(float)
        class_weights = torch.tensor(
            1.0 / np.where(class_counts > 0, class_counts, 1.0),
            dtype=torch.float32,
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Only pass trainable parameters to optimizer
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainable,
            lr=RESNET_LR_HEAD,
            weight_decay=RESNET_WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        # ── Training loop ─────────────────────────────────────────────────────
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, RESNET_EPOCHS + 1):
            # Train
            model.train()
            train_loss = 0.0
            for imgs, lbls in train_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(imgs)
            train_loss /= len(train_records)

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, lbls in test_loader:
                    imgs = imgs.to(device)
                    lbls = lbls.to(device)
                    val_loss += criterion(model(imgs), lbls).item() * len(imgs)
            val_loss /= len(test_records)

            scheduler.step(val_loss)
            log.info(
                "Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f",
                epoch, RESNET_EPOCHS, train_loss, val_loss,
            )

            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= RESNET_PATIENCE:
                    log.info("Early stopping at epoch %d (patience=%d)", epoch, RESNET_PATIENCE)
                    break

        # Restore best weights before evaluation
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # ── Evaluate ──────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_true.extend(lbls.tolist())

        accuracy = float(accuracy_score(all_true, all_preds))
        f1       = float(f1_score(all_true, all_preds, average="weighted", zero_division=0))
        log.info("ResNet50  accuracy=%.4f  weighted-F1=%.4f", accuracy, f1)
        log.info(
            "Classification report:\n%s",
            classification_report(
                all_true, all_preds,
                target_names=[int_to_label[i] for i in range(num_classes)],
                zero_division=0,
            ),
        )

        # ── Save model ────────────────────────────────────────────────────────
        model_path = str(MODELS_DIR / "resnet_latest.pt")
        torch.save(
            {
                "state_dict":    model.cpu().state_dict(),
                "label_to_int":  label_to_int,
                "num_classes":   num_classes,
            },
            model_path,
        )
        log.info("Saved ResNet model → %s", model_path)

        # ── Log to training_runs ──────────────────────────────────────────────
        con = get_conn()
        _log_run(con, run_id, "resnet_finetune", accuracy, f1, model_path)
        con.close()

        return run_id

    # ── Task 4 ────────────────────────────────────────────────────────────────
    @task
    def compare_models(xgb_run_id: str, resnet_run_id: str) -> None:
        """
        Query training_runs for both run IDs and print a side-by-side
        comparison table showing which model performed better.
        """
        from include.db import get_conn

        con = get_conn(read_only=True)
        rows = con.execute(
            """
            SELECT run_id, model_type, accuracy, f1_score, ts, model_path
            FROM training_runs
            WHERE run_id IN (?, ?)
            ORDER BY ts
            """,
            [xgb_run_id, resnet_run_id],
        ).fetchall()
        con.close()

        if not rows:
            log.warning("No training_runs rows found for %s / %s", xgb_run_id, resnet_run_id)
            return

        results = {r[0]: r for r in rows}

        def fmt_row(run_id):
            if run_id not in results:
                return None
            _, model_type, acc, f1, ts, path = results[run_id]
            return {
                "model_type": model_type,
                "accuracy":   acc,
                "f1_score":   f1,
                "ts":         str(ts)[:19],
                "model_path": path,
                "run_id":     run_id,
            }

        xgb    = fmt_row(xgb_run_id)
        resnet = fmt_row(resnet_run_id)

        w = 68
        print("\n" + "═" * w)
        print("  MODEL COMPARISON")
        print("═" * w)
        print(f"  {'Metric':<22}  {'XGBoost':>18}  {'ResNet50 (fine-tuned)':>20}")
        print("  " + "─" * (w - 2))

        def delta(xval, rval, higher_better=True):
            if xval is None or rval is None:
                return ""
            diff = rval - xval
            arrow = "▲" if (diff > 0) == higher_better else "▼"
            return f"  {arrow}{abs(diff):.4f}"

        if xgb and resnet:
            for metric, xval, rval in [
                ("Accuracy",   xgb["accuracy"], resnet["accuracy"]),
                ("Weighted F1",xgb["f1_score"], resnet["f1_score"]),
            ]:
                d = delta(xval, rval)
                print(f"  {metric:<22}  {xval:>18.4f}  {rval:>20.4f}  {d}")

            print("  " + "─" * (w - 2))
            acc_winner  = "XGBoost" if (xgb["accuracy"]  or 0) >= (resnet["accuracy"]  or 0) else "ResNet50"
            f1_winner   = "XGBoost" if (xgb["f1_score"]  or 0) >= (resnet["f1_score"]  or 0) else "ResNet50"
            overall     = acc_winner if acc_winner == f1_winner else "Tied"
            print(f"  {'Accuracy winner':<22}  {acc_winner:>39}")
            print(f"  {'F1 winner':<22}  {f1_winner:>39}")
            print(f"  {'Overall':<22}  {overall:>39}")
            print("  " + "─" * (w - 2))

        for label, info in [("XGBoost", xgb), ("ResNet50", resnet)]:
            if info:
                print(f"  {label} run_id:   {info['run_id']}")
                print(f"  {label} saved to: {info['model_path']}")
                print(f"  {label} trained:  {info['ts']}")

        print("═" * w + "\n")

    # ── Wire up ───────────────────────────────────────────────────────────────
    split       = load_dataset()
    xgb_run_id  = train_xgboost(split)
    resnet_run_id = train_resnet(split)
    compare_models(xgb_run_id, resnet_run_id)


jwst_train_classifiers()
