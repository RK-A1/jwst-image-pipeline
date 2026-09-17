# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local data pipeline on macOS (M4 MacBook Air) that ingests James Webb Space Telescope photos from Flickr, stores metadata and image embeddings in DuckDB, and trains image classifiers. Orchestrated by Apache Airflow via Astro CLI.

## Common Commands

```bash
# Start/stop local Airflow (Docker-based)
astro dev start
astro dev stop

# Trigger DAGs manually
astro dev run airflow dags trigger jwst_flickr_ingest
astro dev run airflow dags trigger jwst_feature_extraction
astro dev run airflow dags trigger jwst_train_classifiers

# Airflow UI: http://localhost:8080  (admin / admin)

# Tag consolidation (runs outside Airflow, on host)
python include/tag_consolidation.py            # dry-run
python include/tag_consolidation.py --apply    # write labels to DB

# Streamlit explorer app
streamlit run app.py

# Query DuckDB directly
duckdb include/jwst.duckdb "SELECT count(*) FROM photos"
```

There are no tests in this project.

## Architecture

```
Flickr API → jwst_flickr_ingest DAG → DuckDB (photos table) + include/images/
                                            ↓
                              jwst_feature_extraction DAG → embeddings in DuckDB
                                            ↓
                         tag_consolidation.py → canonical_label in DuckDB
                                            ↓
                         jwst_train_classifiers DAG → models/ + training_runs table
                                            ↓
                     predict_labels (end of ingest DAG) → predicted_label in DuckDB
```

### Key modules

- **`include/db.py`** — DuckDB connection helper (`get_conn()`) and `init_schema()`. All DB access goes through this. `DB_PATH` points to `include/jwst.duckdb`.
- **`include/tag_consolidation.py`** — Maps Flickr tags to canonical labels using ordered `TAG_RULES`. First matching rule wins. Run as a standalone script, not an Airflow task.
- **`app.py`** — Streamlit explorer (read-only DB connection). Four pages: Overview, Photo Browser, Similarity Search, Model Performance.

### DAGs (`dags/`)

| DAG | Schedule | Purpose |
|-----|----------|---------|
| `jwst_flickr_ingest` | `@daily` | Fetch all Flickr IDs, diff against DB, download + upsert new photos, predict labels |
| `jwst_feature_extraction` | `@daily` | ResNet50 embeddings for photos missing them (dynamic task mapping, batches of 32) |
| `jwst_train_classifiers` | Manual | Parallel XGBoost + fine-tuned ResNet50 training → compare_models |

### DuckDB Schema

Two tables: `photos` (photo_id PK, title, description, tags TEXT[], image_path, date_taken, embedding FLOAT[] 2048-dim, canonical_label, predicted_label) and `training_runs` (run_id PK, ts, model_type, accuracy, f1_score, model_path).

## Key Constraints

- **Astro CLI mounts only `dags/`, `plugins/`, `include/`, `tests/`** — all data (DB, images, models, torch cache) lives under `include/` so it persists to the host filesystem.
- **DuckDB is single-writer.** Use `get_conn(read_only=True)` for reads; serialize all writes through one task at a time.
- **`docker-compose.override.yml` does NOT work with Astro CLI** — Astro generates its compose config internally.
- **MPS backend for PyTorch** — always check `torch.backends.mps.is_available()` before falling back to CPU. DataLoader must use `num_workers=0` on macOS/MPS.
- **Flickr API** — use `flickr.urls.lookupUser` with the path alias URL (not `flickr.people.findByUsername`). Rate limit: 3600 req/hr; DAG sleeps 0.3s between requests. Ingest DAG diffs all Flickr IDs against DB (no date-based watermark — Flickr `date_taken` is unreliable).
- **DB image_path convention** — paths stored in DuckDB use the Docker-internal prefix (`/usr/local/airflow/include/images/`). The Streamlit app remaps via `local_image_path()` to the host path.
- **Predictions below 0.6 confidence** are stored as `unclassified`.
- **`PIL.Image.MAX_IMAGE_PIXELS = None`** must be set before opening JWST images (they exceed the default decompression bomb limit).
- **ResNet fine-tuning** freezes layers 1-3, only trains layer4 + fc head. Uses differential LRs (1e-5 backbone / 1e-4 head) and class-weight balancing.

## Environment

Flickr API key in `.env` at project root (loaded automatically by Astro CLI):
```
FLICKR_API_KEY=<your_key>
```
`.env` is gitignored. Never commit it.

## Claude Code Instructions

- Never add `Co-Authored-By: Claude` or any Claude authorship attribution to git commits.
