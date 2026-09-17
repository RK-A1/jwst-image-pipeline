# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

An Airflow (Astro CLI) pipeline that ingests the NASA Webb Flickr account into DuckDB,
embeds images with ResNet50, labels them with Claude, and publishes a curated dataset
of real observations to `include/data/dataset/`. It is a portfolio project: code
quality, tests and clear engineering decisions matter as much as the output.

## Commands

```bash
astro dev start                      # Airflow UI at http://localhost:8080
astro dev stop
astro dev pytest                     # tests inside the runtime image (includes torch)
astro dev run dags trigger jwst_dataset

.venv/bin/pytest                     # tests on the host; torch test skips
duckdb include/data/warehouse/jwst.duckdb "SELECT count(*) FROM photos"
python scripts/compare_label_runs.py # diff two label runs after a codebook change
```

## Layout

- `dags/` — orchestration only. Tasks import from `include.jwst_pipeline` inside the
  task function so DAG parsing stays fast.
- `include/jwst_pipeline/` — all logic. Must never import Airflow, except `assets.py`.
- `include/data/` — all state. Only `dataset/` outputs and `labels/raw_labels.jsonl`
  are tracked in git; the warehouse and the 18 GB of images are local.
- `tests/` — `conftest.py` points `JWST_DATA_DIR` at a temp dir for every test.

## Rules that are easy to break

- **Astro mounts only `dags/`, `plugins/`, `include/`, `tests/`.** State written
  anywhere else vanishes with the container. `docker-compose.override.yml` did not
  work with this Astro setup.
- **`include/data` must stay in `.dockerignore`.** Otherwise the 18 GB image archive
  goes into the Docker build context.
- **DuckDB is single-writer, across processes.** Any task that opens the warehouse
  needs `pool=WAREHOUSE_POOL`; `tests/test_dags.py` enforces this. Do not hold the pool
  across network calls: split fetching and writing into separate tasks.
- **Never call a paid API in tests or without being asked.** `jwst_label` is
  manual-trigger only for this reason. Flickr calls also need the user's go-ahead.
- **The label log is append-only and is the system of record.** Never rewrite or
  truncate `raw_labels.jsonl`; relabel by appending, since the latest record per
  photo wins.
- **`codebook.md` and `include/jwst_pipeline/codebook.py` must agree.** Change both,
  bump `CODEBOOK_VERSION`, and run `tests/test_codebook.py`.
- **Keep `strict: True` on the labelling tool.** Without it, enum values are advisory.
- **Never put a requests exception message in a log or result without redacting it.**
  Flickr's API key is a query parameter.
- **`PIL.Image.MAX_IMAGE_PIXELS = None`** is needed for JWST originals; use
  `images.open_rgb` / `images.verify` rather than opening images directly.
- **The DuckDB version is pinned** in both requirements files, because host and
  container must read the same storage format.

## Environment

`.env` at the project root (gitignored, loaded by Astro): `FLICKR_API_KEY`,
`ANTHROPIC_API_KEY`. See `.env.example`.

## Git

- Never add `Co-Authored-By: Claude` or any Claude attribution to commits or PRs.
  Commits are authored by the repository owner only.
