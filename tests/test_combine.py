"""The combine step: order-independent join, NFC, missing/extra accounting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import (Dict, List)

from thai_asr_batch.combine import (CombineStats, combine, combine_low_memory)
from thai_asr_batch.config import AppConfig

from conftest import (THAI_COMPOSED, THAI_DECOMPOSED)


def _write_predictions(path: Path, rows: List[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{sid}\t{text}" for sid, text in rows) + "\n", encoding="utf-8"
    )
    return path


def _read_rows(path: Path) -> List[List[str]]:
    lines: List[str] = path.read_text(encoding="utf-8").splitlines()
    return [line.split("\t") for line in lines]


def test_combined_file_has_the_required_columns(
    app_config: AppConfig, sample_tsv: Path, tmp_path: Path
) -> None:
    predicted: Path = _write_predictions(
        tmp_path / "predicted-sample.tsv",
        [("100-100000-0", "p0"), ("100-100000-1", "p1"), ("100-100000-2", "p2"),
         ("148-148801-10", "p10"), ("148-148801-11", "p11")],
    )
    output: Path = tmp_path / "combined.tsv"
    stats: CombineStats = combine(sample_tsv, predicted, output, app_config)

    rows: List[List[str]] = _read_rows(output)
    assert rows[0] == ["segment_id", "orig_text", "pred_text"]
    assert stats.matched == 5
    assert stats.missing_prediction == 0


def test_join_is_order_independent(
    app_config: AppConfig, sample_tsv: Path, tmp_path: Path
) -> None:
    """Predictions shuffled relative to input must still line up."""
    predicted: Path = _write_predictions(
        tmp_path / "predicted-sample.tsv",
        [("148-148801-11", "p11"), ("100-100000-2", "p2"), ("100-100000-0", "p0"),
         ("148-148801-10", "p10"), ("100-100000-1", "p1")],
    )
    output: Path = tmp_path / "combined.tsv"
    combine(sample_tsv, predicted, output, app_config)

    rows: List[List[str]] = _read_rows(output)[1:]
    mapping: Dict[str, str] = {row[0]: row[2] for row in rows}
    assert mapping["100-100000-0"] == "p0"
    assert mapping["148-148801-11"] == "p11"
    # Input order is preserved even though predictions were shuffled.
    assert [row[0] for row in rows][0] == "100-100000-0"


def test_sara_am_spellings_match_after_combining(
    app_config: AppConfig, tmp_path: Path
) -> None:
    """The motivating case: reference and prediction differ only in สระอำ encoding.

    After NFKC the two columns are identical, so WER sees a match rather than
    a spurious error.
    """
    source: Path = tmp_path / "in.tsv"
    source.write_text(f"100-100000-0\t{THAI_DECOMPOSED}\n", encoding="utf-8")
    predicted: Path = _write_predictions(
        tmp_path / "predicted-in.tsv", [("100-100000-0", THAI_COMPOSED)]
    )
    output: Path = tmp_path / "combined.tsv"
    combine(source, predicted, output, app_config)

    row: List[str] = _read_rows(output)[1]
    assert row[1] == row[2], "NFKC should make the two สระอำ spellings identical"


def test_canonical_composition_is_unified(
    app_config: AppConfig, tmp_path: Path
) -> None:
    """A genuine canonical difference collapses too."""
    source: Path = tmp_path / "canon.tsv"
    source.write_text("100-100000-0\té\n", encoding="utf-8")
    predicted: Path = _write_predictions(
        tmp_path / "predicted-canon.tsv", [("100-100000-0", "é")]
    )
    output: Path = tmp_path / "combined.tsv"
    combine(source, predicted, output, app_config)

    row: List[str] = _read_rows(output)[1]
    assert row[1] == row[2] == "é"


def test_missing_predictions_become_empty_rows(
    app_config: AppConfig, sample_tsv: Path, tmp_path: Path
) -> None:
    """The combined file stays 1:1 with input so WER scoring can align."""
    predicted: Path = _write_predictions(
        tmp_path / "predicted-sample.tsv", [("100-100000-0", "p0")]
    )
    output: Path = tmp_path / "combined.tsv"
    stats: CombineStats = combine(sample_tsv, predicted, output, app_config)

    rows: List[List[str]] = _read_rows(output)[1:]
    assert len(rows) == 5
    assert stats.missing_prediction == 4
    assert rows[1][2] == ""


def test_extra_predictions_are_counted(
    app_config: AppConfig, sample_tsv: Path, tmp_path: Path
) -> None:
    predicted: Path = _write_predictions(
        tmp_path / "predicted-sample.tsv",
        [("100-100000-0", "p0"), ("999-999999-9", "orphan")],
    )
    output: Path = tmp_path / "combined.tsv"
    stats: CombineStats = combine(sample_tsv, predicted, output, app_config)
    assert stats.extra_prediction == 1


def test_duplicate_predictions_take_the_last(
    app_config: AppConfig, sample_tsv: Path, tmp_path: Path
) -> None:
    """A resumed run may re-transcribe rows whose checkpoint never fsynced."""
    predicted: Path = _write_predictions(
        tmp_path / "predicted-sample.tsv",
        [("100-100000-0", "first"), ("100-100000-0", "second")],
    )
    output: Path = tmp_path / "combined.tsv"
    combine(sample_tsv, predicted, output, app_config)
    rows: List[List[str]] = _read_rows(output)[1:]
    assert rows[0][2] == "second"


def test_jsonl_output_uses_the_same_keys(
    app_config: AppConfig, sample_tsv: Path, tmp_path: Path
) -> None:
    predicted: Path = _write_predictions(
        tmp_path / "predicted-sample.tsv", [("100-100000-0", "p0")]
    )
    output: Path = tmp_path / "combined.jsonl"
    combine(sample_tsv, predicted, output, app_config, output_format="jsonl")

    first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert set(first) == {"segment_id", "orig_text", "pred_text"}


def test_low_memory_join_matches_the_hash_join(
    app_config: AppConfig, sample_tsv: Path, tmp_path: Path
) -> None:
    app_config.paths.scratch_dir.mkdir(parents=True, exist_ok=True)
    predicted: Path = _write_predictions(
        tmp_path / "predicted-sample.tsv",
        [("148-148801-11", "p11"), ("100-100000-0", "p0")],
    )
    hash_out: Path = tmp_path / "hash.tsv"
    merge_out: Path = tmp_path / "merge.tsv"

    combine(sample_tsv, predicted, hash_out, app_config)
    combine_low_memory(sample_tsv, predicted, merge_out, app_config)

    hash_map: Dict[str, str] = {r[0]: r[2] for r in _read_rows(hash_out)[1:]}
    merge_map: Dict[str, str] = {r[0]: r[2] for r in _read_rows(merge_out)[1:]}
    assert hash_map == merge_map


def test_missing_report_is_written(
    app_config: AppConfig, sample_tsv: Path, tmp_path: Path
) -> None:
    predicted: Path = _write_predictions(
        tmp_path / "predicted-sample.tsv", [("100-100000-0", "p0")]
    )
    report: Path = tmp_path / "missing.txt"
    combine(sample_tsv, predicted, tmp_path / "c.tsv", app_config, missing_report=report)
    assert report.exists()
    assert len(report.read_text(encoding="utf-8").strip().splitlines()) == 4
