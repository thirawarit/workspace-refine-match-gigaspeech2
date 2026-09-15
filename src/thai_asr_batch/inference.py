"""Inference orchestrator: read -> resume-filter -> bucket -> transcribe -> persist."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import (Iterable, Iterator, Optional)

from .audio import (AudioError, AudioProbe, ensure_conforming, estimate_duration,
                    exceeds_window, probe_audio)
from .batching import (BatchOutcome, DeadContextError, SystematicFailureError,
                       WorkItem, bucketed_batches, transcribe_with_isolation)
from .checkpoint import CheckpointStore
from .config import (AppConfig, resolve_device)
from .io_formats import (Format, detect_format, open_reader, open_writer,
                         predicted_output_path, truncate_to_last_complete_line)
from .logging_utils import log_clip_failure
from .model import AsrModelWrapper
from .records import (Record, normalize_text)

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass
class InferenceStats:
    total_seen: int = 0
    skipped_resume: int = 0
    skipped_missing_audio: int = 0
    succeeded: int = 0
    failed: int = 0
    audio_seconds: float = 0.0
    elapsed_seconds: float = 0.0

    @property
    def realtime_factor(self) -> float:
        if self.elapsed_seconds <= 0.0:
            return 0.0
        return self.audio_seconds / self.elapsed_seconds

    def summary(self) -> str:
        return (
            f"seen={self.total_seen} ok={self.succeeded} failed={self.failed} "
            f"skipped_resume={self.skipped_resume} "
            f"missing_audio={self.skipped_missing_audio} "
            f"elapsed={self.elapsed_seconds:.1f}s rtf={self.realtime_factor:.1f}x"
        )


@dataclass
class _PreparedItem:
    record: Record
    probe: AudioProbe
    duration: float


def run_inference(
    input_path: Path,
    cfg: AppConfig,
    limit: Optional[int] = None,
    dry_run: bool = False,
    resume: bool = True,
    force_resume: bool = False,
) -> InferenceStats:
    """Transcribe ``input_path`` into ``predicted-<name>`` beside it."""
    started: float = time.monotonic()
    stats: InferenceStats = InferenceStats()

    fmt: Format = detect_format(input_path)
    output_path: Path = predicted_output_path(input_path, cfg.paths.output_dir)
    LOGGER.info("input=%s format=%s", input_path, fmt)
    LOGGER.info("output=%s", output_path)

    store: CheckpointStore = CheckpointStore(
        root=cfg.paths.checkpoint_dir,
        input_path=input_path,
        cfg=cfg.checkpoint,
        config_fingerprint=cfg.fingerprint(),
        input_format=fmt,
    )
    if resume and cfg.checkpoint.enabled:
        store.open(force=force_resume)
    else:
        LOGGER.info("resume disabled; starting from scratch")

    records: Iterator[Record] = _iter_records(input_path, cfg, store, resume, stats, limit)

    if dry_run:
        _dry_run(records, cfg, stats)
        stats.elapsed_seconds = time.monotonic() - started
        LOGGER.info("dry-run complete: %s", stats.summary())
        return stats

    device: str = resolve_device(cfg.device)
    model: AsrModelWrapper = AsrModelWrapper(cfg.model, device, audio_cfg=cfg.audio)
    model.load()

    append: bool = resume and cfg.checkpoint.enabled and output_path.exists()
    if append:
        # A hard kill can leave a half-written final line here too.
        truncate_to_last_complete_line(output_path)

    writer = open_writer(output_path, fmt, append=append)
    consecutive_failures: int = 0
    last_systematic: Optional[str] = None
    systematic_streak: int = 0

    try:
        prepared: Iterator[WorkItem] = _prepare_items(records, cfg, stats)
        for batch in bucketed_batches(prepared, cfg.batch):
            outcome: BatchOutcome = transcribe_with_isolation(model, batch)
            succeeded = outcome.succeeded
            failed = outcome.failed

            # A misconfiguration fails every clip with the same message and will
            # never succeed, so stop on the second such batch instead of
            # grinding through the corpus writing empty predictions.
            if outcome.systematic_error is not None:
                if outcome.systematic_error == last_systematic:
                    systematic_streak += 1
                else:
                    last_systematic = outcome.systematic_error
                    systematic_streak = 1
                if systematic_streak >= 2:
                    raise SystematicFailureError(
                        f"{systematic_streak} consecutive batches failed with an "
                        f"identical error, so this is a configuration fault, not "
                        f"bad audio:\n  {outcome.systematic_error}\n"
                        "Nothing was written for these clips. Fix the cause and "
                        "re-run; completed segments resume from the checkpoint."
                    )
            else:
                last_systematic = None
                systematic_streak = 0

            if outcome.whole_batch_failed:
                consecutive_failures += 1
                if consecutive_failures >= cfg.runtime.max_consecutive_batch_failures:
                    raise DeadContextError(
                        f"{consecutive_failures} consecutive whole-batch failures; "
                        "aborting rather than writing empty predictions. The CUDA "
                        "context is likely unusable — restart the run to resume."
                    )
            else:
                consecutive_failures = 0

            for item in batch:
                segment_id: str = item.record.segment_id
                if segment_id in succeeded:
                    writer.write(item.record, normalize_text(succeeded[segment_id]))
                    stats.succeeded += 1
                    stats.audio_seconds += item.duration_estimate
                else:
                    reason: str = failed.get(segment_id, "unknown failure")
                    log_clip_failure(segment_id, item.audio_path, reason)
                    writer.write(item.record, "")
                    stats.failed += 1
                store.mark_done(segment_id)

            if store.should_flush():
                # Order matters: predictions durable before the checkpoint that
                # claims them, else a crash between the two loses rows silently.
                writer.flush()
                store.flush()

            processed: int = stats.succeeded + stats.failed
            if processed and processed % cfg.runtime.progress_every_n == 0:
                _log_progress(stats, started)
    finally:
        writer.close()
        store.close()

    stats.elapsed_seconds = time.monotonic() - started
    LOGGER.info("inference complete: %s", stats.summary())
    return stats


def _iter_records(
    input_path: Path,
    cfg: AppConfig,
    store: CheckpointStore,
    resume: bool,
    stats: InferenceStats,
    limit: Optional[int],
) -> Iterator[Record]:
    reader = open_reader(input_path, cfg.paths.audio_root)
    emitted: int = 0

    for record in reader:
        stats.total_seen += 1

        if resume and cfg.checkpoint.enabled and store.is_done(record.segment_id):
            stats.skipped_resume += 1
            continue

        yield record
        emitted += 1
        if limit is not None and emitted >= limit:
            LOGGER.info("--limit %d reached; stopping intake", limit)
            return


def _prepare_items(
    records: Iterable[Record],
    cfg: AppConfig,
    stats: InferenceStats,
) -> Iterator[WorkItem]:
    """Probe audio on a thread pool so the GPU never stalls on stat() calls."""
    workers: int = max(1, cfg.runtime.io_workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for prepared in pool.map(lambda record: _prepare_one(record, cfg), records):
            if prepared is None:
                stats.skipped_missing_audio += 1
                continue
            yield WorkItem(
                record=prepared.record,
                audio_path=prepared.probe.path,
                duration_estimate=prepared.duration,
            )


def _prepare_one(record: Record, cfg: AppConfig) -> Optional[_PreparedItem]:
    audio_path: Optional[Path] = record.audio_filepath
    if audio_path is None:
        log_clip_failure(record.segment_id, Path(""), "no audio path resolved")
        return None

    probe: AudioProbe = probe_audio(audio_path)
    if not probe.is_readable:
        log_clip_failure(record.segment_id, audio_path, probe.error or "unreadable")
        return None

    try:
        resolved, _converted = ensure_conforming(
            audio_path, cfg.audio, cfg.paths.scratch_dir, probe=probe
        )
    except AudioError as exc:
        log_clip_failure(record.segment_id, audio_path, str(exc))
        return None

    duration: float = estimate_duration(resolved, probe=probe if resolved == audio_path else None)

    # Whisper keeps only the first encoder window and truncates the rest with no
    # error, so a long clip returns partial text that reads like a bad
    # transcription. Flag it; still transcribe, so the row is not lost.
    if exceeds_window(duration, cfg.audio):
        log_clip_failure(
            record.segment_id,
            audio_path,
            f"clip is {duration:.1f}s, longer than the "
            f"{cfg.audio.max_duration_seconds:.0f}s window; only the first "
            "window will be transcribed",
        )

    final_probe: AudioProbe = probe if resolved == audio_path else probe_audio(resolved)
    return _PreparedItem(record=record, probe=final_probe, duration=duration)


def _dry_run(records: Iterable[Record], cfg: AppConfig, stats: InferenceStats) -> None:
    """Validate audio paths without loading the model.

    This is how a 10M-row corpus gets checked on a laptop before GPU days are
    committed to it.
    """
    non_conforming: int = 0
    over_window: int = 0
    for record in records:
        audio_path: Optional[Path] = record.audio_filepath
        if audio_path is None:
            stats.skipped_missing_audio += 1
            continue
        probe: AudioProbe = probe_audio(audio_path)
        if probe.is_readable and exceeds_window(
            probe.duration_seconds or 0.0, cfg.audio
        ):
            over_window += 1
        if not probe.is_readable:
            stats.skipped_missing_audio += 1
            LOGGER.debug("%s: %s", record.segment_id, probe.error)
            continue
        if probe.sample_rate != cfg.audio.target_sample_rate or probe.channels != 1:
            non_conforming += 1
        stats.succeeded += 1
        stats.audio_seconds += probe.duration_seconds or 0.0

    if non_conforming:
        LOGGER.warning(
            "%d file(s) need resampling (will convert on the fly)", non_conforming
        )
    if over_window:
        # Reported up front so the real exposure is known before GPU time is spent.
        LOGGER.warning(
            "%d clip(s) exceed the %.0fs window and will be truncated to the "
            "first window",
            over_window, cfg.audio.max_duration_seconds,
        )


def _log_progress(stats: InferenceStats, started: float) -> None:
    elapsed: float = time.monotonic() - started
    processed: int = stats.succeeded + stats.failed
    rate: float = processed / elapsed if elapsed > 0 else 0.0
    rtf: float = stats.audio_seconds / elapsed if elapsed > 0 else 0.0
    LOGGER.info(
        "progress: %d done (%d failed) | %.1f clips/s | %.1fx realtime | %.1fh audio",
        processed, stats.failed, rate, rtf, stats.audio_seconds / 3600.0,
    )
