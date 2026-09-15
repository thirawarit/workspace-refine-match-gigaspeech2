"""Logging setup: Bangkok-timezone formatter, session log file, error journal."""

from __future__ import annotations

import json
import logging
import sys
from datetime import (datetime, timezone)
from pathlib import Path
from typing import (Any, Dict, Optional)
from zoneinfo import ZoneInfo

BANGKOK_TZ: ZoneInfo = ZoneInfo("Asia/Bangkok")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
ERROR_LOGGER_NAME: str = "thai_asr_batch.errors"


class BangkokFormatter(logging.Formatter):
    """Formatter that renders timestamps in Asia/Bangkok regardless of host TZ.

    The VPS is most likely UTC, so relying on the host clock would silently
    produce non-Bangkok timestamps. Overriding formatTime pins it explicitly.
    """

    def formatTime(  # noqa: N802 - stdlib-defined name
        self,
        record: logging.LogRecord,
        datefmt: Optional[str] = None,
    ) -> str:
        moment: datetime = datetime.fromtimestamp(record.created, tz=timezone.utc)
        local: datetime = moment.astimezone(BANGKOK_TZ)
        if datefmt:
            return local.strftime(datefmt)
        return local.strftime(DATE_FORMAT)


def session_stamp(now: Optional[datetime] = None) -> str:
    """Return a filename-safe Bangkok timestamp, e.g. ``20260915-134500``."""
    moment: datetime = now if now is not None else datetime.now(tz=BANGKOK_TZ)
    return moment.astimezone(BANGKOK_TZ).strftime("%Y%m%d-%H%M%S")


def setup_logging(
    log_dir: Path,
    level: int = logging.INFO,
    stamp: Optional[str] = None,
) -> Path:
    """Configure root logging to stdout plus ``log_dir/log-<stamp>.txt``.

    Returns the path of the session log file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    resolved_stamp: str = stamp if stamp is not None else session_stamp()
    log_path: Path = log_dir / f"log-{resolved_stamp}.txt"

    formatter: BangkokFormatter = BangkokFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    root: logging.Logger = logging.getLogger()
    root.setLevel(level)
    for existing in list(root.handlers):
        root.removeHandler(existing)

    stream_handler: logging.StreamHandler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    file_handler: logging.FileHandler = logging.FileHandler(
        log_path, mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    _setup_error_journal(log_dir, resolved_stamp)
    return log_path


def _setup_error_journal(log_dir: Path, stamp: str) -> Path:
    """Attach a machine-readable JSONL journal for per-clip failures.

    At 10M clips even a 0.01% failure rate is ~1000 entries, so failures need to
    be greppable as data, not only as prose in the session log.
    """
    journal_path: Path = log_dir / f"errors-{stamp}.jsonl"
    error_logger: logging.Logger = logging.getLogger(ERROR_LOGGER_NAME)
    error_logger.setLevel(logging.ERROR)
    error_logger.propagate = False
    for existing in list(error_logger.handlers):
        error_logger.removeHandler(existing)

    handler: logging.FileHandler = logging.FileHandler(
        journal_path, mode="a", encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    error_logger.addHandler(handler)
    return journal_path


def log_clip_failure(segment_id: str, audio_path: Path, reason: str) -> None:
    """Record a single failed clip as one JSON object on its own line."""
    payload: Dict[str, Any] = {
        "ts": datetime.now(tz=BANGKOK_TZ).strftime(DATE_FORMAT),
        "segment_id": segment_id,
        "audio_path": str(audio_path),
        "reason": reason,
    }
    logging.getLogger(ERROR_LOGGER_NAME).error(
        json.dumps(payload, ensure_ascii=False)
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
