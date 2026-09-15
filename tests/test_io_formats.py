"""TSV/JSONL readers and writers, naming, and partial-line recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from thai_asr_batch.io_formats import (JsonlReader, TsvReader, TsvWriter,
                                       detect_format, open_writer,
                                       predicted_output_path,
                                       truncate_to_last_complete_line)
from thai_asr_batch.records import Record


def test_predicted_naming_matches_the_requirement(tmp_path: Path) -> None:
    assert predicted_output_path(Path("AbC.tsv"), tmp_path).name == "predicted-AbC.tsv"
    assert predicted_output_path(Path("x.jsonl"), tmp_path).name == "predicted-x.jsonl"


def test_tsv_reader_parses_headerless_rows(sample_tsv: Path, audio_root: Path) -> None:
    records: List[Record] = list(TsvReader(sample_tsv, audio_root))
    assert len(records) == 5
    assert records[0].segment_id == "100-100000-0"
    assert "COVID-19" in records[0].text
    assert records[-1].segment_id == "148-148801-11"


def test_tsv_reader_derives_audio_paths(sample_tsv: Path, audio_root: Path) -> None:
    records: List[Record] = list(TsvReader(sample_tsv, audio_root))
    expected: Path = audio_root / "train" / "148" / "148801" / "148-148801-11.wav"
    assert records[-1].audio_filepath == expected


def test_tsv_reader_skips_blank_and_tabless_rows(tmp_path: Path, audio_root: Path) -> None:
    path: Path = tmp_path / "messy.tsv"
    path.write_text(
        "\n"
        "100-100000-0\tgood\n"
        "no-tab-here-at-all\n"
        "   \n"
        "148-148801-11\talso good\n",
        encoding="utf-8",
    )
    records: List[Record] = list(TsvReader(path, audio_root))
    assert [r.segment_id for r in records] == ["100-100000-0", "148-148801-11"]


def test_tsv_reader_keeps_empty_text(tmp_path: Path, audio_root: Path) -> None:
    """A blank reference is legitimate; the clip still needs a prediction."""
    path: Path = tmp_path / "empty.tsv"
    path.write_text("100-100000-0\t\n", encoding="utf-8")
    records: List[Record] = list(TsvReader(path, audio_root))
    assert len(records) == 1
    assert records[0].text == ""


def test_tsv_reader_splits_only_on_the_first_tab(tmp_path: Path, audio_root: Path) -> None:
    path: Path = tmp_path / "multi.tsv"
    path.write_text("100-100000-0\ta\tb\n", encoding="utf-8")
    records: List[Record] = list(TsvReader(path, audio_root))
    assert records[0].text == "a\tb"


def test_tsv_reader_strips_bom(tmp_path: Path, audio_root: Path) -> None:
    path: Path = tmp_path / "bom.tsv"
    path.write_text("﻿100-100000-0\ttext\n", encoding="utf-8")
    records: List[Record] = list(TsvReader(path, audio_root))
    assert records[0].segment_id == "100-100000-0"


def test_tsv_reader_skips_malformed_ids(tmp_path: Path, audio_root: Path) -> None:
    path: Path = tmp_path / "bad.tsv"
    path.write_text("nope\ttext\n100-100000-0\tgood\n", encoding="utf-8")
    records: List[Record] = list(TsvReader(path, audio_root))
    assert [r.segment_id for r in records] == ["100-100000-0"]


def test_jsonl_reader_derives_segment_id_from_stem(
    sample_jsonl: Path, audio_root: Path
) -> None:
    records: List[Record] = list(JsonlReader(sample_jsonl, audio_root))
    assert [r.segment_id for r in records] == ["100-100000-0", "148-148801-11"]
    assert records[0].duration == 0.1


def test_jsonl_reader_skips_invalid_json(tmp_path: Path, audio_root: Path) -> None:
    path: Path = tmp_path / "bad.jsonl"
    path.write_text('{"audio_filepath": "/a/100-100000-0.wav", "text": "ok"}\nnot json\n',
                    encoding="utf-8")
    records: List[Record] = list(JsonlReader(path, audio_root))
    assert len(records) == 1


def test_detect_format_trusts_content_over_suffix(tmp_path: Path) -> None:
    misnamed: Path = tmp_path / "actually.tsv"
    misnamed.write_text('{"audio_filepath": "/a/b.wav", "text": "x"}\n', encoding="utf-8")
    assert detect_format(misnamed) == "jsonl"


def test_detect_format_plain_tsv(sample_tsv: Path) -> None:
    assert detect_format(sample_tsv) == "tsv"


def test_tsv_round_trip(tmp_path: Path, audio_root: Path) -> None:
    out: Path = tmp_path / "predicted-x.tsv"
    writer = open_writer(out, "tsv")
    writer.write(Record(segment_id="100-100000-0", text=""), "ผลลัพธ์")
    writer.close()

    records: List[Record] = list(TsvReader(out, audio_root))
    assert records[0].segment_id == "100-100000-0"
    assert records[0].text == "ผลลัพธ์"


def test_tsv_writer_neutralizes_embedded_tabs(tmp_path: Path) -> None:
    out: Path = tmp_path / "p.tsv"
    writer: TsvWriter = TsvWriter(out, append=False)
    writer.write(Record(segment_id="100-100000-0", text=""), "a\tb\nc")
    writer.close()
    line: str = out.read_text(encoding="utf-8").rstrip("\n")
    assert line.count("\t") == 1


def test_jsonl_writer_preserves_manifest_fields(tmp_path: Path) -> None:
    out: Path = tmp_path / "p.jsonl"
    writer = open_writer(out, "jsonl")
    writer.write(
        Record(segment_id="s", text="orig", audio_filepath=Path("/a/s.wav"), duration=1.5),
        "predicted",
    )
    writer.close()
    payload = json.loads(out.read_text(encoding="utf-8").strip())
    assert payload["audio_filepath"] == "/a/s.wav"
    assert payload["text"] == "predicted"
    assert payload["duration"] == 1.5


def test_truncate_drops_a_partial_final_line(tmp_path: Path) -> None:
    """A kill mid-append must lose the partial id, not keep it.

    Keeping a truncated id would permanently skip a real segment; dropping it
    costs one free re-transcription.
    """
    path: Path = tmp_path / "shard.done"
    path.write_text("100-100000-0\n100-100000-1\n100-1000", encoding="utf-8")
    removed: int = truncate_to_last_complete_line(path)
    assert removed == len("100-1000")
    assert path.read_text(encoding="utf-8") == "100-100000-0\n100-100000-1\n"


def test_truncate_is_a_noop_on_clean_files(tmp_path: Path) -> None:
    path: Path = tmp_path / "clean.done"
    path.write_text("100-100000-0\n", encoding="utf-8")
    assert truncate_to_last_complete_line(path) == 0
    assert path.read_text(encoding="utf-8") == "100-100000-0\n"


def test_truncate_handles_a_single_partial_line(tmp_path: Path) -> None:
    path: Path = tmp_path / "only.done"
    path.write_text("100-1000", encoding="utf-8")
    truncate_to_last_complete_line(path)
    assert path.read_text(encoding="utf-8") == ""


def test_truncate_on_missing_file_is_safe(tmp_path: Path) -> None:
    assert truncate_to_last_complete_line(tmp_path / "nope.done") == 0
