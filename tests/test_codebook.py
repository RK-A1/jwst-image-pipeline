"""
codebook.md is the specification and codebook.py is what the model is sent. The rules
live in both, so these tests fail when an enum value is added to one and not the other.
"""

import re

from include.jwst_pipeline import codebook
from include.jwst_pipeline.config import PROJECT_ROOT


def _section(number: int) -> str:
    text = (PROJECT_ROOT / "codebook.md").read_text()
    match = re.search(rf"^## {number}\. .*?(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    assert match, f"codebook.md has no section {number}"
    return match.group(0)


def _table_values(number: int) -> set[str]:
    return set(re.findall(r"^\| `([^`]+)` \|", _section(number), re.MULTILINE))


def _without_not_applicable(values: list[str]) -> set[str]:
    return set(values) - {codebook.NOT_APPLICABLE}


def test_gate_categories_match_codebook_md():
    assert _table_values(1) == set(codebook.GATE_CATEGORIES)


def test_subjects_match_codebook_md():
    assert _table_values(2) == _without_not_applicable(codebook.SUBJECTS)


def test_modalities_match_codebook_md():
    assert _table_values(3) == _without_not_applicable(codebook.MODALITIES)


def test_instruments_match_codebook_md():
    line = next(l for l in _section(5).splitlines() if l.startswith("One of:"))
    assert set(re.findall(r"`([^`]+)`", line)) == set(codebook.INSTRUMENTS)


def test_prompt_names_every_enum_value():
    values = (codebook.GATE_CATEGORIES + codebook.SUBJECTS
              + codebook.MODALITIES + codebook.INSTRUMENTS)
    missing = [v for v in values if v not in codebook.SYSTEM]
    assert not missing, f"the prompt never mentions {missing}"


def test_schema_is_valid_for_strict_tool_use():
    schema = codebook.SCHEMA
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
