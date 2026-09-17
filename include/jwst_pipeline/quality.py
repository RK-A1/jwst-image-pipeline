"""
Data quality checks.

A check is a query returning the number of offending rows. Zero passes. A failing
`error` check raises DataQualityError and fails the task, which stops anything
downstream from consuming bad data; a failing `warn` check is logged and passes.
Checks are plain SQL so each one reads as the rule it enforces.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import duckdb

from include.jwst_pipeline import codebook

log = logging.getLogger(__name__)


class DataQualityError(RuntimeError):
    pass


@dataclass(frozen=True)
class Check:
    name: str
    query: str | Callable[[duckdb.DuckDBPyConnection], int]
    severity: str = "error"   # "error" fails the task, "warn" only logs


@dataclass(frozen=True)
class Result:
    check: Check
    violations: int

    @property
    def failed(self) -> bool:
        return self.violations > 0 and self.check.severity == "error"


def run(con: duckdb.DuckDBPyConnection, checks: list[Check], *, context: str) -> list[Result]:
    results = []
    for check in checks:
        if callable(check.query):
            n = int(check.query(con))
        else:
            n = int(con.execute(check.query).fetchone()[0])
        results.append(Result(check, n))
        if n == 0:
            log.info("[%s] ok    %s", context, check.name)
        elif check.severity == "error":
            log.error("[%s] FAIL  %s: %d rows", context, check.name, n)
        else:
            log.warning("[%s] warn  %s: %d rows", context, check.name, n)

    failed = [r for r in results if r.failed]
    if failed:
        summary = "; ".join(f"{r.check.name} ({r.violations})" for r in failed)
        raise DataQualityError(f"{context}: {len(failed)} check(s) failed: {summary}")
    return results


def _in(values: list[str]) -> str:
    return ", ".join(f"'{v}'" for v in values)


PHOTO_CHECKS = [
    Check(
        "downloaded photos record their file size",
        "SELECT count(*) FROM photos WHERE image_file IS NOT NULL AND image_bytes IS NULL",
    ),
    Check(
        "image files are named after their photo",
        "SELECT count(*) FROM photos WHERE image_file IS NOT NULL "
        "AND image_file <> photo_id || '.jpg'",
    ),
    Check(
        "downloads still failing after 3 attempts",
        "SELECT count(*) FROM photos WHERE image_file IS NULL AND download_attempts >= 3",
        severity="warn",
    ),
]

EMBEDDING_CHECKS = [
    Check(
        "every embedding belongs to a known photo",
        "SELECT count(*) FROM embeddings ANTI JOIN photos USING (photo_id)",
    ),
    Check(
        "embeddings are finite",
        "SELECT count(*) FROM embeddings "
        "WHERE len(list_filter(vector::FLOAT[], x -> isnan(x) OR isinf(x))) > 0",
    ),
    Check(
        "embeddings are not all zero",
        "SELECT count(*) FROM embeddings WHERE list_max(vector::FLOAT[]) = 0",
    ),
    Check(
        "images that could not be embedded",
        "SELECT count(*) FROM embedding_errors",
        severity="warn",
    ),
]

LABEL_CHECKS = [
    Check(
        "every label belongs to a known photo",
        "SELECT count(*) FROM labels ANTI JOIN photos USING (photo_id)",
    ),
    Check(
        "gate_category is a codebook value",
        f"SELECT count(*) FROM labels WHERE gate_category NOT IN ({_in(codebook.GATE_CATEGORIES)})",
    ),
    Check(
        "observations have a subject and modality",
        f"SELECT count(*) FROM labels WHERE gate_category = '{codebook.KEEP_GATE}' "
        f"AND (subject = '{codebook.NOT_APPLICABLE}' OR modality = '{codebook.NOT_APPLICABLE}')",
    ),
]


def dataset_checks(expected_labels: int, published_rows: int | None, allow_shrink: bool) -> list[Check]:
    """
    Audit a staged dataset before it is published. Runs against views named `kept`
    and `rejected` over the staged files.
    """
    checks = [
        Check("the dataset is not empty",
              "SELECT (count(*) = 0)::INT FROM kept"),
        Check("photo_id is unique",
              "SELECT count(*) - count(DISTINCT photo_id) FROM kept"),
        Check("no photo is both kept and rejected",
              "SELECT count(*) FROM kept SEMI JOIN rejected USING (photo_id)"),
        Check("kept and rejected together account for every label",
              f"SELECT abs((SELECT count(*) FROM kept) + (SELECT count(*) FROM rejected) "
              f"- {int(expected_labels)})"),
        Check("subject is a codebook value",
              f"SELECT count(*) FROM kept WHERE subject NOT IN "
              f"({_in([s for s in codebook.SUBJECTS if s != codebook.NOT_APPLICABLE])})"),
        Check("modality is a codebook value",
              f"SELECT count(*) FROM kept WHERE modality NOT IN "
              f"({_in([m for m in codebook.MODALITIES if m != codebook.NOT_APPLICABLE])})"),
        Check("instrument is a codebook value",
              f"SELECT count(*) FROM kept WHERE instrument NOT IN ({_in(codebook.INSTRUMENTS)})"),
        Check("every release group has exactly one primary",
              "SELECT count(*) FROM (SELECT release_group FROM kept GROUP BY 1 "
              "HAVING count(*) FILTER (WHERE is_primary) <> 1)"),
        Check("object names not found verbatim in the caption",
              "SELECT count(*) FROM kept WHERE NOT object_name_verified", severity="warn"),
        Check("rows without an embedding",
              "SELECT count(*) FROM kept WHERE NOT has_embedding", severity="warn"),
        Check("rows without a downscaled image",
              "SELECT count(*) FROM kept WHERE image_file IS NULL", severity="warn"),
    ]
    if published_rows is not None:
        checks.append(Check(
            f"row count has not dropped below the published {published_rows}",
            f"SELECT greatest({int(published_rows)} - count(*), 0) FROM kept",
            severity="warn" if allow_shrink else "error",
        ))
    return checks
