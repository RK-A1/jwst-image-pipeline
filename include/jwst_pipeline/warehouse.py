"""
DuckDB warehouse: connections, schema and migrations.

Tables
  photos            one row per Flickr photo: caption metadata, download state, and the
                    label from the retired tag matcher where one exists
  embeddings        one ResNet50 vector per photo; the column type fixes it at 2048 dims
  embedding_errors  images that failed to decode, so they are not retried on every run
  labels            model labels, loaded from the append-only label log (labelling.py)

Migrations are numbered and append-only. migrate() applies whatever a database is
missing, each in its own transaction, and records it in schema_migrations.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

import duckdb

from include.jwst_pipeline.config import paths

log = logging.getLogger(__name__)

EMBEDDING_DIM = 2048

MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "initial schema", f"""
        CREATE TABLE photos (
            photo_id            VARCHAR PRIMARY KEY,
            title               VARCHAR,
            description         VARCHAR,
            tags                VARCHAR[],
            date_taken          TIMESTAMP,   -- uploader-supplied, sometimes wildly wrong
            first_seen_at       TIMESTAMP NOT NULL,
            last_seen_at        TIMESTAMP,   -- last ingest that found it on Flickr
            image_file          VARCHAR,     -- under the images dir; NULL until downloaded
            image_bytes         BIGINT,
            downloaded_at       TIMESTAMP,
            download_attempts   INTEGER NOT NULL DEFAULT 0,
            last_download_error VARCHAR,
            legacy_tag_label    VARCHAR      -- keyword label from the retired tag matcher
        );

        CREATE TABLE embeddings (
            photo_id    VARCHAR PRIMARY KEY,
            model       VARCHAR NOT NULL,
            vector      FLOAT[{EMBEDDING_DIM}] NOT NULL,
            created_at  TIMESTAMP NOT NULL
        );

        CREATE TABLE embedding_errors (
            photo_id    VARCHAR PRIMARY KEY,
            error       VARCHAR NOT NULL,
            failed_at   TIMESTAMP NOT NULL
        );

        CREATE TABLE labels (
            photo_id          VARCHAR PRIMARY KEY,
            gate_category     VARCHAR NOT NULL,
            subject           VARCHAR NOT NULL,
            modality          VARCHAR NOT NULL,
            object_name       VARCHAR,
            instrument        VARCHAR NOT NULL,
            confidence        VARCHAR NOT NULL,
            rationale         VARCHAR,
            model             VARCHAR NOT NULL,
            backend           VARCHAR,
            vision            BOOLEAN,
            codebook_version  VARCHAR NOT NULL,
            seconds           DOUBLE,
            prompt_tokens     INTEGER,
            output_tokens     INTEGER,
            cache_read_tokens INTEGER
        );
    """),
]


def connect(read_only: bool = False, *, wait_s: float = 120.0) -> duckdb.DuckDBPyConnection:
    """
    Open the warehouse, waiting out a lock held by another process.

    Airflow tasks never contend, because the warehouse pool serialises them. The wait
    covers everything outside Airflow, such as a duckdb shell left open on the host.
    """
    path = paths().warehouse
    if read_only and not path.exists():
        raise FileNotFoundError(f"no warehouse at {path}; run the ingest DAG first")
    path.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + wait_s
    delay = 0.5
    while True:
        try:
            return duckdb.connect(str(path), read_only=read_only)
        except duckdb.IOException as exc:
            if "lock" not in str(exc).lower() or time.monotonic() + delay > deadline:
                raise
            log.warning("warehouse locked by another process, retrying in %.1fs", delay)
            time.sleep(delay)
            delay = min(delay * 2, 10.0)


@contextmanager
def session(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    con = connect(read_only)
    try:
        yield con
    finally:
        con.close()


def migrate(con: duckdb.DuckDBPyConnection) -> list[int]:
    """Apply pending migrations in order. Returns the versions applied."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     INTEGER PRIMARY KEY,
            name        VARCHAR NOT NULL,
            applied_at  TIMESTAMP NOT NULL DEFAULT current_timestamp
        )
    """)
    applied = {v for (v,) in con.execute("SELECT version FROM schema_migrations").fetchall()}

    done = []
    for version, name, sql in MIGRATIONS:
        if version in applied:
            continue
        con.begin()
        try:
            con.execute(sql)
            con.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (?, ?)", [version, name]
            )
            con.commit()
        except Exception:
            con.rollback()
            raise
        log.info("applied migration %d: %s", version, name)
        done.append(version)
    return done
