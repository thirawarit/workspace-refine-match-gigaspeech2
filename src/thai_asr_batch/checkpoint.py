"""Sharded, crash-safe resume state.

A flat done-file with 10M lines is too slow to re-scan at every restart. State
is sharded by the first ``segment_id`` component, which is also the first audio
directory level, so checkpoint locality matches on-disk locality and each shard
loads roughly once per run.

Plain append-only text, not SQLite: it survives ``kill -9`` predictably and can
be inspected with ``wc -l`` while a multi-day job is running.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import (Dict, Optional, Set, TextIO)

from .config import CheckpointConfig
from .io_formats import truncate_to_last_complete_line
from .records import (is_valid_segment_id, shard_of)

LOGGER: logging.Logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
MANIFEST_NAME: str = "manifest.json"


class CheckpointMismatchError(RuntimeError):
    """Raised when existing checkpoint state does not match the current input."""


class CheckpointStore:
    """Append-only per-shard record of completed segment ids."""

    def __init__(
        self,
        root: Path,
        input_path: Path,
        cfg: CheckpointConfig,
        config_fingerprint: str,
        input_format: str,
    ) -> None:
        self.dir: Path = root / input_path.stem
        self.cfg: CheckpointConfig = cfg
        self.input_path: Path = input_path
        self.config_fingerprint: str = config_fingerprint
        self.input_format: str = input_format

        self._handles: Dict[str, TextIO] = {}
        self._cache: "OrderedDict[str, Set[str]]" = OrderedDict()
        self._pending: int = 0
        self._last_flush: float = time.monotonic()
        self._completed_delta: int = 0

    # -- lifecycle ---------------------------------------------------------

    def open(self, force: bool = False) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._verify_manifest(force=force)

    def _manifest_payload(self) -> Dict[str, object]:
        stat: os.stat_result = self.input_path.stat()
        return {
            "schema_version": SCHEMA_VERSION,
            "input_path": str(self.input_path.resolve()),
            "input_size": stat.st_size,
            "input_mtime": int(stat.st_mtime),
            "input_format": self.input_format,
            "config_fingerprint": self.config_fingerprint,
        }

    def _verify_manifest(self, force: bool) -> None:
        path: Path = self.dir / MANIFEST_NAME
        current: Dict[str, object] = self._manifest_payload()

        if not path.exists():
            path.write_text(json.dumps(current, indent=2), encoding="utf-8")
            return

        try:
            previous: Dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("unreadable checkpoint manifest (%s); rewriting", exc)
            path.write_text(json.dumps(current, indent=2), encoding="utf-8")
            return

        differences: list[str] = [
            f"{key}: checkpoint={previous.get(key)!r} current={value!r}"
            for key, value in current.items()
            if previous.get(key) != value
        ]
        if not differences:
            return

        message: str = (
            "checkpoint state does not match this input:\n  "
            + "\n  ".join(differences)
        )
        if not force:
            # Resuming against a changed input would interleave two corpora.
            raise CheckpointMismatchError(
                message + "\nRe-run with --force-resume to override, or clear "
                f"{self.dir} to start fresh."
            )
        LOGGER.warning("%s\n--force-resume given; continuing anyway", message)
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    def close(self) -> None:
        self.flush()
        for handle in self._handles.values():
            if not handle.closed:
                handle.close()
        self._handles.clear()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # -- queries -----------------------------------------------------------

    def _shard_path(self, shard: str) -> Path:
        return self.dir / f"shard-{shard}.done"

    def load_completed_for_shard(self, shard: str) -> Set[str]:
        """Return the done-set for one shard, loading and LRU-caching it."""
        cached: Optional[Set[str]] = self._cache.get(shard)
        if cached is not None:
            self._cache.move_to_end(shard)
            return cached

        done: Set[str] = self._read_shard(shard)
        self._cache[shard] = done
        self._cache.move_to_end(shard)
        while len(self._cache) > max(1, self.cfg.max_cached_shards):
            evicted, _ = self._cache.popitem(last=False)
            LOGGER.debug("evicted shard %s from resume cache", evicted)
        return done

    def _read_shard(self, shard: str) -> Set[str]:
        path: Path = self._shard_path(shard)
        if not path.exists():
            return set()

        truncate_to_last_complete_line(path)

        done: Set[str] = set()
        malformed: int = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                candidate: str = line.strip()
                if not candidate:
                    continue
                if not is_valid_segment_id(candidate):
                    malformed += 1
                    continue
                done.add(candidate)
        if malformed:
            LOGGER.warning("%s: dropped %d malformed line(s)", path, malformed)
        LOGGER.debug("shard %s: %d completed segment(s)", shard, len(done))
        return done

    def is_done(self, segment_id: str) -> bool:
        if not self.cfg.enabled:
            return False
        return segment_id in self.load_completed_for_shard(shard_of(segment_id))

    def count_completed(self) -> int:
        """Total completed ids across all shards (for ``status``)."""
        total: int = 0
        for path in sorted(self.dir.glob("shard-*.done")):
            with path.open("rb") as handle:
                total += sum(1 for _ in handle)
        return total

    # -- writes ------------------------------------------------------------

    def _handle_for(self, shard: str) -> TextIO:
        handle: Optional[TextIO] = self._handles.get(shard)
        if handle is not None and not handle.closed:
            return handle
        path: Path = self._shard_path(shard)
        path.parent.mkdir(parents=True, exist_ok=True)
        truncate_to_last_complete_line(path)
        opened: TextIO = path.open("a", encoding="utf-8", newline="\n")
        self._handles[shard] = opened
        return opened

    def mark_done(self, segment_id: str) -> None:
        if not self.cfg.enabled:
            return
        shard: str = shard_of(segment_id)
        self._handle_for(shard).write(f"{segment_id}\n")
        cached: Optional[Set[str]] = self._cache.get(shard)
        if cached is not None:
            cached.add(segment_id)
        self._pending += 1
        self._completed_delta += 1

    def should_flush(self) -> bool:
        if self._pending <= 0:
            return False
        if self._pending >= self.cfg.flush_every_n:
            return True
        return (time.monotonic() - self._last_flush) >= self.cfg.flush_every_seconds

    def flush(self) -> None:
        """fsync every open shard handle.

        Callers MUST flush the predictions writer first — see
        :meth:`io_formats._BaseWriter.flush`.
        """
        for handle in self._handles.values():
            if not handle.closed:
                handle.flush()
                os.fsync(handle.fileno())
        self._pending = 0
        self._last_flush = time.monotonic()
