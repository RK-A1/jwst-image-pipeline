import duckdb
import pytest

from include.jwst_pipeline.quality import Check, DataQualityError, run


@pytest.fixture
def mem():
    c = duckdb.connect()
    c.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1), (2), (2)) v(x)")
    return c


def test_passing_and_warning_checks_do_not_raise(mem):
    results = run(mem, [
        Check("no negatives", "SELECT count(*) FROM t WHERE x < 0"),
        Check("duplicates", "SELECT count(*) - count(DISTINCT x) FROM t", severity="warn"),
    ], context="test")
    assert [r.violations for r in results] == [0, 1]
    assert not any(r.failed for r in results)


def test_error_checks_raise_after_running_every_check(mem):
    seen = []

    def counting(con):
        seen.append(True)
        return 0

    with pytest.raises(DataQualityError, match=r"unique x \(1\)"):
        run(mem, [
            Check("unique x", "SELECT count(*) - count(DISTINCT x) FROM t"),
            Check("callable check", counting),
        ], context="test")
    assert seen == [True]
