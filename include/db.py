"""
DuckDB connection helper and schema initialization.

Usage:
    from include.db import get_conn, init_schema

    con = get_conn()          # read-write connection
    con = get_conn(read_only=True)  # safe for parallel reads
"""

from pathlib import Path
import duckdb

DB_PATH = Path(__file__).resolve().parents[1] / "include" / "jwst.duckdb"


def get_conn(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection to the project database.

    DuckDB allows multiple read-only connections but only one read-write
    connection at a time. Serialize all writes through a single Airflow task
    (or use read_only=True for query-only workloads).
    """
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def init_schema() -> None:
    """Create tables if they do not already exist."""
    con = get_conn()
    con.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            photo_id        TEXT PRIMARY KEY,
            title           TEXT,
            description     TEXT,
            tags            TEXT[],           -- list of tag strings
            image_path      TEXT,             -- absolute path to local file
            date_taken      TIMESTAMP,
            date_ingested   TIMESTAMP DEFAULT current_timestamp,
            embedding       FLOAT[],          -- ResNet feature vector (2048-dim)
            canonical_label TEXT,             -- tag-based label set by tag_consolidation.py
            predicted_label TEXT
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id      TEXT PRIMARY KEY,
            ts          TIMESTAMP DEFAULT current_timestamp,
            model_type  TEXT,                 -- 'xgboost' or 'resnet_finetune'
            accuracy    DOUBLE,
            f1_score    DOUBLE,
            model_path  TEXT                  -- path to saved checkpoint / model file
        )
    """)

    con.close()


if __name__ == "__main__":
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_schema()
    print(f"Schema initialised at {DB_PATH}")
