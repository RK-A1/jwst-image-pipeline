import json
from types import SimpleNamespace

import pytest

from conftest import label_record
from include.jwst_pipeline import codebook, labelling


def write_log(path, *records, raw_tail: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + raw_tail)


def test_caption_is_cleaned_before_it_is_truncated(monkeypatch):
    monkeypatch.setattr(labelling, "DESC_CHARS", 20)
    desc = '<a href="https://example.com/very/long/link" rel="nofollow">Cassiopeia A</a> &amp; more text'
    assert labelling.user_text("T", desc) == "TITLE: T\n\nCAPTION: Cassiopeia A & more "


def test_validate_catches_the_silent_rejection_bug():
    # A modality value in gate_category once turned five real observations into rejections.
    assert labelling.validate(label_record("1", gate_category="image")) == ["gate_category='image'"]
    assert labelling.validate(label_record("1", subject="not_applicable"))
    assert labelling.validate(label_record("1")) == []
    assert labelling.validate(label_record("1", gate_category="people_event",
                                           subject="not_applicable", modality="not_applicable")) == []


def test_read_log_skips_an_incomplete_final_line_only(data_dir):
    path = data_dir / "labels" / "raw_labels.jsonl"
    write_log(path, label_record("1"), raw_tail='{"photo_id": "2", "gate')
    assert [r["photo_id"] for r in labelling.read_log(path)] == ["1"]

    path.write_text('{"broken\n' + json.dumps(label_record("1")) + "\n")
    with pytest.raises(labelling.LabelLogError, match="line 1"):
        labelling.read_log(path)


def test_load_log_keeps_the_latest_record_per_photo(con, add_photo, data_dir):
    add_photo("1")
    path = data_dir / "labels" / "raw_labels.jsonl"
    write_log(path, label_record("1", subject="nebula"), label_record("1", subject="planetary_nebula"))
    assert labelling.load_log(con, path) == 1
    assert con.execute("SELECT subject FROM labels").fetchone()[0] == "planetary_nebula"


def test_invalid_log_leaves_the_labels_table_untouched(con, add_photo, data_dir):
    add_photo("1")
    path = data_dir / "labels" / "raw_labels.jsonl"
    write_log(path, label_record("1"))
    labelling.load_log(con, path)

    write_log(path, label_record("1"), label_record("2", gate_category="comparison_composite"))
    with pytest.raises(labelling.LabelLogError, match="violate the codebook"):
        labelling.load_log(con, path)
    assert con.execute("SELECT photo_id FROM labels").fetchall() == [("1",)]


def test_queue_selects_only_photos_that_still_need_a_label(con, add_photo, data_dir):
    add_photo("labelled")
    add_photo("in_log_not_loaded")
    add_photo("pre_launch", date_taken="2019-06-01 00:00:00")
    add_photo("not_downloaded", with_file=False)
    add_photo("future_date", date_taken="2124-04-30 18:00:00")   # Flickr really has these
    add_photo("no_date", date_taken=None)
    add_photo("launch_day", date_taken="2021-12-25 00:00:00")

    path = data_dir / "labels" / "raw_labels.jsonl"
    write_log(path, label_record("labelled"))
    labelling.load_log(con, path)
    labelling.append_log(path, label_record("in_log_not_loaded"))

    queued = [r[0] for r in labelling.queue(con, path, limit=100)]
    assert queued == ["future_date", "launch_day", "no_date"]
    assert len(labelling.queue(con, path, limit=2)) == 2


def test_call_api_request_shape_and_usage():
    captured = {}

    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", input={"gate_category": "people_event"})],
                usage=SimpleNamespace(input_tokens=3400, output_tokens=150, cache_read_input_tokens=0),
                stop_reason="tool_use",
            )

    label, usage = labelling.call_api(SimpleNamespace(messages=Messages()),
                                      "claude-haiku-4-5", "Title", "Caption", "BASE64")
    assert label == {"gate_category": "people_event"}
    assert usage == {"prompt_tokens": 3400, "output_tokens": 150, "cache_read_tokens": 0}

    [tool] = captured["tools"]
    assert tool["strict"] is True and tool["input_schema"] is codebook.SCHEMA
    assert captured["tool_choice"] == {"type": "tool", "name": tool["name"]}
    content = captured["messages"][0]["content"]
    assert [block["type"] for block in content] == ["image", "text"]


def test_build_record_stamps_the_codebook_version():
    record = labelling.build_record("1", "t", {"subject": "star"}, {"prompt_tokens": 1},
                                    model="m", backend="api", vision=True, seconds=1.234)
    assert record["codebook_version"] == codebook.CODEBOOK_VERSION
    assert record["seconds"] == 1.23
