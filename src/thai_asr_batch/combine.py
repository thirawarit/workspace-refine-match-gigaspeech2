"""Join the input file with its predictions on ``segment_id``."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (Dict, Iterator, Optional, Set, TextIO)

from .config import AppConfig
from .io_formats import (COMBINED_HEADER, Format, TSV_DELIMITER, detect_format,
                         open_reader)
from .records import (Record, normalize_text)

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CombineStats:
    total_input: int
    matched: int
    missing_prediction: int
    extra_prediction: int

    def summary(self) -> str:
        return (
            f"input={self.total_input} matched={self.matched} "
            f"missing_pred={self.missing_prediction} extra_pred={self.extra_prediction}"
        )


def combine(
    input_path: Path,
    predicted_path: Path,
    output_path: Path,
    cfg: AppConfig,
    output_format: Optional[Format] = None,
    missing_report: Optional[Path] = None,
) -> CombineStats:
    """Write ``segment_id``/``orig_text``/``pred_text``, joined by segment id.

    Predictions are the smaller side, so they form the hash and the input is
    streamed. At 10M rows that is roughly 2-3 GB — comfortable on the 503 GB
    VPS. Use ``--low-memory`` on constrained hosts.
    """
    fmt: Format = output_format if output_format is not None else detect_format(input_path)
    predictions: Dict[str, str] = _load_predictions(predicted_path, cfg)
    LOGGER.info("loaded %d prediction(s) from %s", len(predictions), predicted_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: Set[str] = set()
    total: int = 0
    matched: int = 0
    missing: int = 0
    missing_ids: list[str] = []

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        if fmt == "tsv":
            # Named columns imply a header; unlike the headerless input, this
            # file is for human and eval consumption.
            handle.write(TSV_DELIMITER.join(COMBINED_HEADER) + "\n")

        for record in open_reader(input_path, cfg.paths.audio_root):
            total += 1
            segment_id: str = record.segment_id
            seen.add(segment_id)

            pred_text: Optional[str] = predictions.get(segment_id)
            if pred_text is None:
                missing += 1
                missing_ids.append(segment_id)
                pred_text = ""
            else:
                matched += 1

            _write_row(handle, fmt, segment_id, normalize_text(record.text),
                       normalize_text(pred_text))

    extra: int = len(set(predictions) - seen)
    if extra:
        LOGGER.warning(
            "%d prediction(s) have no matching input row; possible checkpoint/input mismatch",
            extra,
        )
    if missing_ids:
        report: Path = missing_report or (cfg.paths.log_dir / f"missing-{input_path.stem}.txt")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(missing_ids) + "\n", encoding="utf-8")
        LOGGER.warning("%d input row(s) lack predictions; ids in %s", missing, report)

    stats: CombineStats = CombineStats(
        total_input=total,
        matched=matched,
        missing_prediction=missing,
        extra_prediction=extra,
    )
    LOGGER.info("combine complete: %s -> %s", stats.summary(), output_path)
    return stats


def _write_row(
    handle: TextIO,
    fmt: Format,
    segment_id: str,
    orig_text: str,
    pred_text: str,
) -> None:
    if fmt == "jsonl":
        handle.write(
            json.dumps(
                {"segment_id": segment_id, "orig_text": orig_text, "pred_text": pred_text},
                ensure_ascii=False,
            )
            + "\n"
        )
        return
    safe_orig: str = orig_text.replace(TSV_DELIMITER, " ").replace("\n", " ")
    safe_pred: str = pred_text.replace(TSV_DELIMITER, " ").replace("\n", " ")
    handle.write(f"{segment_id}{TSV_DELIMITER}{safe_orig}{TSV_DELIMITER}{safe_pred}\n")


def _load_predictions(predicted_path: Path, cfg: AppConfig) -> Dict[str, str]:
    """Build segment_id -> pred_text from the interim file."""
    predictions: Dict[str, str] = {}
    for record in open_reader(predicted_path, cfg.paths.audio_root):
        # Later duplicates win: a resumed run may re-transcribe a few segments
        # whose predictions were written but whose checkpoint never fsynced.
        predictions[record.segment_id] = record.text
    return predictions


def combine_low_memory(
    input_path: Path,
    predicted_path: Path,
    output_path: Path,
    cfg: AppConfig,
    output_format: Optional[Format] = None,
) -> CombineStats:
    """Sort-merge join for hosts that cannot hold the prediction map in RAM.

    Slower than :func:`combine` but bounded in memory; this should be the
    default if combine ever runs on the laptop rather than the VPS.
    """
    import tempfile

    fmt: Format = output_format if output_format is not None else detect_format(input_path)

    with tempfile.TemporaryDirectory(dir=str(cfg.paths.scratch_dir)) as tmp:
        tmp_dir: Path = Path(tmp)
        sorted_input: Path = _sort_to_tempfile(
            input_path, tmp_dir / "input.sorted", cfg
        )
        sorted_pred: Path = _sort_to_tempfile(
            predicted_path, tmp_dir / "pred.sorted", cfg
        )
        return _merge_join(sorted_input, sorted_pred, output_path, fmt, cfg)


def _sort_to_tempfile(source: Path, target: Path, cfg: AppConfig) -> Path:
    import subprocess

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        rows: Iterator[Record] = open_reader(source, cfg.paths.audio_root)
        proc = subprocess.Popen(
            ["sort", "-t", TSV_DELIMITER, "-k", "1,1"],
            stdin=subprocess.PIPE,
            stdout=handle,
            text=True,
            encoding="utf-8",
        )
        assert proc.stdin is not None
        for record in rows:
            safe: str = record.text.replace(TSV_DELIMITER, " ").replace("\n", " ")
            proc.stdin.write(f"{record.segment_id}{TSV_DELIMITER}{safe}\n")
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"sort failed for {source}")
    return target


def _merge_join(
    sorted_input: Path,
    sorted_pred: Path,
    output_path: Path,
    fmt: Format,
    cfg: AppConfig,
) -> CombineStats:
    total: int = 0
    matched: int = 0
    missing: int = 0
    extra: int = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sorted_input.open("r", encoding="utf-8") as left, \
         sorted_pred.open("r", encoding="utf-8") as right, \
         output_path.open("w", encoding="utf-8", newline="\n") as out:

        if fmt == "tsv":
            out.write(TSV_DELIMITER.join(COMBINED_HEADER) + "\n")

        pred_line: str = right.readline()
        for input_line in left:
            input_id, _, orig_text = input_line.rstrip("\n").partition(TSV_DELIMITER)
            total += 1

            while pred_line:
                pred_id, _, _pred_text = pred_line.rstrip("\n").partition(TSV_DELIMITER)
                if pred_id < input_id:
                    extra += 1
                    pred_line = right.readline()
                    continue
                break

            pred_text: str = ""
            if pred_line:
                pred_id, _, candidate = pred_line.rstrip("\n").partition(TSV_DELIMITER)
                if pred_id == input_id:
                    pred_text = candidate
                    matched += 1
                    pred_line = right.readline()
                else:
                    missing += 1
            else:
                missing += 1

            _write_row(out, fmt, input_id, normalize_text(orig_text), normalize_text(pred_text))

        while pred_line:
            extra += 1
            pred_line = right.readline()

    stats: CombineStats = CombineStats(
        total_input=total,
        matched=matched,
        missing_prediction=missing,
        extra_prediction=extra,
    )
    LOGGER.info("combine (low-memory) complete: %s", stats.summary())
    return stats
