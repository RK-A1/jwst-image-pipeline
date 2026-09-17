import duckdb
import pytest

from include.jwst_pipeline import warehouse
from include.jwst_pipeline.config import utcnow


def test_migrate_is_idempotent(con):
    assert warehouse.migrate(con) == []
    versions = con.execute("SELECT version FROM schema_migrations").fetchall()
    assert versions == [(v,) for v, _, _ in warehouse.MIGRATIONS]


def test_failed_migration_rolls_back(con, monkeypatch):
    monkeypatch.setattr(warehouse, "MIGRATIONS", warehouse.MIGRATIONS + [
        (99, "broken", "CREATE TABLE half_done (x INT); SELECT * FROM no_such_table;"),
    ])
    with pytest.raises(duckdb.Error):
        warehouse.migrate(con)
    tables = {t for (t,) in con.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
    assert "half_done" not in tables
    assert con.execute("SELECT count(*) FROM schema_migrations WHERE version = 99").fetchone()[0] == 0


def test_embedding_column_rejects_wrong_dimension(con, add_photo):
    add_photo("1")
    with pytest.raises(duckdb.Error):
        con.execute("INSERT INTO embeddings VALUES ('1', 'm', ?, ?)", [[0.1] * 10, utcnow()])


def test_read_only_connection_needs_an_existing_warehouse(data_dir):
    with pytest.raises(FileNotFoundError):
        warehouse.connect(read_only=True)


def test_connect_waits_out_a_lock_then_gives_up(con, monkeypatch):
    calls = []

    def locked(*args, **kwargs):
        calls.append(1)
        raise duckdb.IOException("Could not set lock on file: Conflicting lock is held")

    monkeypatch.setattr(warehouse.duckdb, "connect", locked)
    monkeypatch.setattr(warehouse.time, "sleep", lambda s: None)
    with pytest.raises(duckdb.IOException):
        warehouse.connect(wait_s=0.1)
    assert len(calls) >= 1
