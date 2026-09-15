"""Readers and writers for headerless TSV and NeMo-manifest JSONL.

Both formats normalize to :class:`~thai_asr_batch.records.Record`, so every
downstream stage is format-agnostic and keyed on ``segment_id``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import TracebackType
from typing import (Any, Dict, Iterator, Literal, Optional, Protocol, TextIO, Type)

from .records import (MalformedSegmentIdError, Record, audio_path_for_segment,
                      segment_id_from_path)

LOGGER: logging.Logger = logging.getLogger(__name__)

Format = Literal["tsv", "jsonl"]

TSV_DELIMITER: str = "\t"
BOM: str = "﻿"
PREDICTED_PREFIX: str = "predicted-"
COMBINED_HEADER: tuple[str, str, str] = ("segment_id", "orig_text", "pred_text")


class RecordReader(Protocol):
    """Streaming source of records. Implementations must be generators."""

    def __iter__(self) -> Iterator[Record]: ...


class RecordWriter(Protocol):
    """Incremental sink for predictions."""

    def write(self, record: Record, pred_text: str) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


def detect_format(path: Path) -> Format:
    """Infer the format from the suffix, then sniff the first non-empty line.

    The sniff catches misnamed files, which matters because a JSONL misread as
    TSV would silently produce one giant malformed segment id per line.
    """
    suffix: str = path.suffix.lower()
    sniffed: Optional[Format] = _sniff_format(path)
    if sniffed is not None:
        if suffix in (".tsv", ".jsonl") and sniffed != _suffix_format(suffix):
            LOGGER.warning(
                "%s has suffix %s but content looks like %s; trusting content",
                path, suffix, sniffed,
            )
        return sniffed
    return _suffix_format(suffix)


def _suffix_format(suffix: str) -> Format:
    return "jsonl" if suffix.lower() == ".jsonl" else "tsv"


def _sniff_format(path: Path) -> Optional[Format]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped: str = line.lstrip(BOM).strip()
                if not stripped:
                    continue
                return "jsonl" if stripped.startswith("{") else "tsv"
    except OSError:
        return None
    return None


class TsvReader:
    """Headerless two-column TSV: ``segment_id \\t text``."""

    def __init__(self, path: Path, audio_root: Path) -> None:
        self.path: Path = path
        self.audio_root: Path = audio_root

    def __iter__(self) -> Iterator[Record]:
        with self.path.open("r", encoding="utf-8") as handle:
            for lineno, raw_line in enumerate(handle, start=1):
                line: str = raw_line.rstrip("\n").rstrip("\r")
                if lineno == 1:
                    line = line.lstrip(BOM)
                if not line.strip():
                    continue

                if TSV_DELIMITER not in line:
                    LOGGER.warning(
                        "%s:%d has no tab separator; skipping", self.path, lineno
                    )
                    continue

                # maxsplit=1: Thai text will not contain tabs, but a stray one
                # must not shift the text into a third phantom column.
                segment_id, text = line.split(TSV_DELIMITER, 1)
                segment_id = segment_id.strip()
                if not segment_id:
                    LOGGER.warning("%s:%d has empty segment id; skipping", self.path, lineno)
                    continue

                try:
                    audio_path: Path = audio_path_for_segment(segment_id, self.audio_root)
                except MalformedSegmentIdError as exc:
                    LOGGER.warning("%s:%d %s; skipping", self.path, lineno, exc)
                    continue

                # An empty text field is legitimate — the reference may be blank
                # but the clip still needs a prediction.
                yield Record(segment_id=segment_id, text=text, audio_filepath=audio_path)


class JsonlReader:
    """NeMo manifest JSONL: ``{"audio_filepath": ..., "text": ..., "duration": ...}``."""

    def __init__(self, path: Path, audio_root: Path) -> None:
        self.path: Path = path
        self.audio_root: Path = audio_root

    def __iter__(self) -> Iterator[Record]:
        with self.path.open("r", encoding="utf-8") as handle:
            for lineno, raw_line in enumerate(handle, start=1):
                line: str = raw_line.strip()
                if lineno == 1:
                    line = line.lstrip(BOM)
                if not line:
                    continue

                try:
                    payload: Dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOGGER.warning("%s:%d invalid JSON (%s); skipping", self.path, lineno, exc)
                    continue

                raw_path: Optional[str] = payload.get("audio_filepath")
                if not raw_path:
                    LOGGER.warning(
                        "%s:%d missing audio_filepath; skipping", self.path, lineno
                    )
                    continue

                audio_path: Path = Path(raw_path)
                if not audio_path.is_absolute():
                    audio_path = self.audio_root / audio_path

                duration_raw: Any = payload.get("duration")
                duration: Optional[float] = (
                    float(duration_raw) if isinstance(duration_raw, (int, float)) else None
                )

                # The manifest path is authoritative; the id derives from it.
                yield Record(
                    segment_id=segment_id_from_path(audio_path),
                    text=str(payload.get("text", "")),
                    audio_filepath=audio_path,
                    duration=duration,
                )


class _BaseWriter:
    def __init__(self, path: Path, append: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode: str = "a" if append else "w"
        self.path: Path = path
        self._handle: TextIO = path.open(mode, encoding="utf-8", newline="\n")

    def flush(self) -> None:
        """Flush and fsync.

        Callers must fsync predictions BEFORE the checkpoint: crashing between
        them re-transcribes a few segments, while the reverse order would mark
        segments done whose predictions were never durable.
        """
        import os

        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if not self._handle.closed:
            self.flush()
            self._handle.close()

    def __enter__(self) -> "_BaseWriter":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()


class TsvWriter(_BaseWriter):
    """Headerless ``segment_id \\t pred_text``, mirroring the input shape."""

    def write(self, record: Record, pred_text: str) -> None:
        safe_text: str = pred_text.replace("\t", " ").replace("\n", " ")
        self._handle.write(f"{record.segment_id}{TSV_DELIMITER}{safe_text}\n")


class JsonlWriter(_BaseWriter):
    """NeMo manifest out, preserving audio_filepath/duration and replacing text."""

    def write(self, record: Record, pred_text: str) -> None:
        payload: Dict[str, Any] = {
            "audio_filepath": str(record.audio_filepath) if record.audio_filepath else "",
            "text": pred_text,
        }
        if record.duration is not None:
            payload["duration"] = record.duration
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def open_reader(path: Path, audio_root: Path, fmt: Optional[Format] = None) -> RecordReader:
    resolved: Format = fmt if fmt is not None else detect_format(path)
    if resolved == "jsonl":
        return JsonlReader(path, audio_root)
    return TsvReader(path, audio_root)


def open_writer(path: Path, fmt: Format, append: bool = False) -> RecordWriter:
    if fmt == "jsonl":
        return JsonlWriter(path, append)
    return TsvWriter(path, append)


def predicted_output_path(input_path: Path, output_dir: Path) -> Path:
    """``AbC.tsv`` -> ``<output_dir>/predicted-AbC.tsv``."""
    return output_dir / f"{PREDICTED_PREFIX}{input_path.name}"


def truncate_to_last_complete_line(path: Path) -> int:
    """Drop a trailing partial line left by a hard kill. Returns bytes removed.

    A truncated final line must be discarded rather than kept: re-transcribing
    one segment is free, whereas retaining a truncated id would permanently and
    silently skip a real segment.
    """
    if not path.exists() or path.stat().st_size == 0:
        return 0

    with path.open("rb+") as handle:
        handle.seek(0, 2)
        size: int = handle.tell()
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return 0

        # Walk back to the last newline.
        position: int = size - 1
        chunk_size: int = 8192
        while position > 0:
            read_start: int = max(0, position - chunk_size)
            handle.seek(read_start)
            chunk: bytes = handle.read(position - read_start)
            index: int = chunk.rfind(b"\n")
            if index != -1:
                new_size: int = read_start + index + 1
                handle.truncate(new_size)
                LOGGER.warning(
                    "%s ended mid-line; truncated %d partial byte(s)",
                    path, size - new_size,
                )
                return size - new_size
            position = read_start

        handle.truncate(0)
        LOGGER.warning("%s contained no complete line; truncated to empty", path)
        return size
