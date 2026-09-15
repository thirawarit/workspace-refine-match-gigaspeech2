"""Length bucketing and per-clip failure isolation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (Dict, Iterable, Iterator, List, Sequence, Tuple)

from .config import BatchConfig
from .model import AsrModelWrapper
from .records import Record

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkItem:
    record: Record
    audio_path: Path
    duration_estimate: float


class DeadContextError(RuntimeError):
    """Raised when too many whole batches fail in a row.

    After a CUDA OOM the context can be left unusable, in which case per-item
    retry also fails and every subsequent batch silently yields empty text. A
    job that stops loudly beats one that writes empty rows for hours.
    """


def bucketed_batches(
    items: Iterable[WorkItem],
    cfg: BatchConfig,
) -> Iterator[List[WorkItem]]:
    """Group items into batches of similar duration to limit padding waste.

    A global sort would need the whole corpus in RAM and would destroy the
    shard locality that resume depends on. Instead this sorts a bounded window
    and emits from it, which captures most of the padding win for free.
    """
    buffer: List[WorkItem] = []
    window: int = max(cfg.batch_size, cfg.sort_buffer_size)

    for item in items:
        buffer.append(item)
        if len(buffer) >= window:
            yield from _drain(buffer, cfg)
            buffer = []

    if buffer:
        yield from _drain(buffer, cfg)


def _drain(buffer: List[WorkItem], cfg: BatchConfig) -> Iterator[List[WorkItem]]:
    buffer.sort(key=lambda item: item.duration_estimate)
    batch: List[WorkItem] = []
    longest: float = 0.0

    for item in buffer:
        prospective_longest: float = max(longest, item.duration_estimate)
        prospective_cost: float = prospective_longest * (len(batch) + 1)

        # Close early when padded cost would blow the cap — this is what keeps
        # an all-long-clip window from OOMing the GPU.
        if batch and prospective_cost > cfg.max_batch_duration_seconds:
            yield batch
            batch = [item]
            longest = item.duration_estimate
            continue

        batch.append(item)
        longest = prospective_longest
        if len(batch) >= cfg.batch_size:
            yield batch
            batch = []
            longest = 0.0

    if batch:
        yield batch


def transcribe_with_isolation(
    model: AsrModelWrapper,
    batch: Sequence[WorkItem],
) -> Tuple[Dict[str, str], Dict[str, str], bool]:
    """Transcribe a batch, isolating failures to the offending clip.

    Returns ``(succeeded, failed, whole_batch_failed)``. On a batch-level
    exception the batch is retried one item at a time so a single poison clip
    does not void its ~15 healthy neighbours. The cost is one wasted pass per
    bad clip, negligible against an expected ~1000 bad clips at 10M scale.
    """
    if not batch:
        return {}, {}, False

    try:
        texts: List[str] = model.transcribe_batch([item.audio_path for item in batch])
        return {item.record.segment_id: text for item, text in zip(batch, texts)}, {}, False
    except Exception as exc:  # noqa: BLE001 - any failure falls back to per-item
        if _is_oom(exc):
            LOGGER.warning("CUDA OOM on batch of %d; clearing cache", len(batch))
            _empty_cuda_cache()
        else:
            LOGGER.warning("batch of %d failed (%s); retrying individually", len(batch), exc)

    succeeded: Dict[str, str] = {}
    failed: Dict[str, str] = {}
    for item in batch:
        try:
            single: List[str] = model.transcribe_batch([item.audio_path])
            succeeded[item.record.segment_id] = single[0] if single else ""
        except Exception as exc:  # noqa: BLE001
            failed[item.record.segment_id] = str(exc)

    whole_batch_failed: bool = not succeeded and bool(failed)
    return succeeded, failed, whole_batch_failed


def _is_oom(exc: BaseException) -> bool:
    text: str = str(exc).lower()
    return "out of memory" in text or exc.__class__.__name__ == "OutOfMemoryError"


def _empty_cuda_cache() -> None:
    # OSError as well as ImportError: a partially extracted CUDA wheel leaves
    # torch importable-looking but its shared libraries unloadable.
    try:
        import torch
    except (ImportError, OSError):
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
