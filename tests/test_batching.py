"""Length bucketing and failure isolation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

from thai_asr_batch.batching import (WorkItem, bucketed_batches,
                                     transcribe_with_isolation)
from thai_asr_batch.config import BatchConfig
from thai_asr_batch.records import Record

from conftest import FakeAsrModel


def _item(segment_id: str, duration: float) -> WorkItem:
    return WorkItem(
        record=Record(segment_id=segment_id, text=""),
        audio_path=Path(f"/audio/{segment_id}.wav"),
        duration_estimate=duration,
    )


def _cfg(batch_size: int = 4, buffer_size: int = 16, cap: float = 1e9) -> BatchConfig:
    return BatchConfig(
        batch_size=batch_size,
        sort_buffer_size=buffer_size,
        max_batch_duration_seconds=cap,
    )


def test_batches_respect_size() -> None:
    items: List[WorkItem] = [_item(f"1-1-{i}", 1.0) for i in range(10)]
    batches: List[List[WorkItem]] = list(bucketed_batches(items, _cfg(batch_size=4)))
    assert [len(b) for b in batches] == [4, 4, 2]


def test_no_item_is_lost_or_duplicated() -> None:
    rng: random.Random = random.Random(1234)
    items: List[WorkItem] = [
        _item(f"1-1-{i}", rng.uniform(0.1, 30.0)) for i in range(257)
    ]
    batches: List[List[WorkItem]] = list(
        bucketed_batches(items, _cfg(batch_size=16, buffer_size=64))
    )
    emitted: List[str] = [item.record.segment_id for batch in batches for item in batch]
    assert sorted(emitted) == sorted(item.record.segment_id for item in items)
    assert len(emitted) == len(set(emitted))


def test_items_within_a_batch_have_similar_durations() -> None:
    durations: List[float] = [30.0, 0.5, 29.0, 0.6, 28.0, 0.7]
    items: List[WorkItem] = [_item(f"1-1-{i}", d) for i, d in enumerate(durations)]
    batches: List[List[WorkItem]] = list(
        bucketed_batches(items, _cfg(batch_size=3, buffer_size=6))
    )
    spreads: List[float] = [
        max(i.duration_estimate for i in b) - min(i.duration_estimate for i in b)
        for b in batches
    ]
    # Bucketing should keep the short clips together and the long ones together.
    assert max(spreads) < 5.0


def test_duration_cap_closes_a_batch_early() -> None:
    items: List[WorkItem] = [_item(f"1-1-{i}", 100.0) for i in range(4)]
    batches: List[List[WorkItem]] = list(
        bucketed_batches(items, _cfg(batch_size=16, buffer_size=16, cap=250.0))
    )
    assert all(
        max(i.duration_estimate for i in b) * len(b) <= 250.0 or len(b) == 1
        for b in batches
    )
    assert len(batches) > 1


def test_empty_input_yields_nothing() -> None:
    assert list(bucketed_batches([], _cfg())) == []


def test_healthy_batch_returns_all_predictions() -> None:
    model: FakeAsrModel = FakeAsrModel()
    batch: List[WorkItem] = [_item("1-1-0", 1.0), _item("1-1-1", 1.0)]
    outcome = transcribe_with_isolation(model, batch)  # type: ignore[arg-type]
    assert set(outcome.succeeded) == {"1-1-0", "1-1-1"}
    assert not outcome.failed
    assert not outcome.whole_batch_failed
    assert outcome.systematic_error is None


def test_one_poison_clip_does_not_void_its_neighbours() -> None:
    """The core isolation guarantee: a bad clip fails alone."""
    model: FakeAsrModel = FakeAsrModel()
    batch: List[WorkItem] = [
        _item("1-1-0", 1.0),
        _item("1-1-boom", 1.0),
        _item("1-1-2", 1.0),
    ]
    outcome = transcribe_with_isolation(model, batch)  # type: ignore[arg-type]
    assert set(outcome.succeeded) == {"1-1-0", "1-1-2"}
    assert set(outcome.failed) == {"1-1-boom"}
    assert not outcome.whole_batch_failed
    assert outcome.systematic_error is None


def test_all_failing_batch_reports_whole_failure() -> None:
    """Signals a possibly dead CUDA context to the orchestrator."""
    model: FakeAsrModel = FakeAsrModel()
    batch: List[WorkItem] = [_item("1-1-boom", 1.0), _item("1-1-boom2", 1.0)]
    outcome = transcribe_with_isolation(model, batch)  # type: ignore[arg-type]
    assert not outcome.succeeded
    assert len(outcome.failed) == 2
    assert outcome.whole_batch_failed


def test_identical_errors_are_flagged_as_systematic() -> None:
    """A misconfiguration fails every clip the same way — not bad audio.

    Models the real case: "Expected 'lang' to be set for AggregateTokenizer."
    on every clip regardless of input.
    """

    class MisconfiguredModel:
        def transcribe_batch(self, audio_paths: List[Path]) -> List[str]:
            raise RuntimeError("Expected 'lang' to be set for AggregateTokenizer.")

    batch: List[WorkItem] = [_item("1-1-0", 1.0), _item("1-1-1", 1.0)]
    outcome = transcribe_with_isolation(MisconfiguredModel(), batch)  # type: ignore[arg-type]
    assert outcome.whole_batch_failed
    assert outcome.systematic_error is not None
    assert "AggregateTokenizer" in outcome.systematic_error


def test_differing_errors_are_not_systematic() -> None:
    """Distinct per-clip failures are bad data, and must not trip the abort."""

    class FlakyModel:
        def transcribe_batch(self, audio_paths: List[Path]) -> List[str]:
            raise RuntimeError(f"corrupt frame in {audio_paths[0].stem}")

    batch: List[WorkItem] = [_item("1-1-0", 1.0), _item("1-1-1", 1.0)]
    outcome = transcribe_with_isolation(FlakyModel(), batch)  # type: ignore[arg-type]
    assert outcome.whole_batch_failed
    assert outcome.systematic_error is None


def test_single_clip_batch_is_never_systematic() -> None:
    """One failing clip is not evidence of a systematic fault."""

    class MisconfiguredModel:
        def transcribe_batch(self, audio_paths: List[Path]) -> List[str]:
            raise RuntimeError("Expected 'lang' to be set for AggregateTokenizer.")

    outcome = transcribe_with_isolation(MisconfiguredModel(), [_item("1-1-0", 1.0)])  # type: ignore[arg-type]
    assert outcome.systematic_error is None


def test_isolation_retries_individually() -> None:
    model: FakeAsrModel = FakeAsrModel()
    batch: List[WorkItem] = [_item("1-1-0", 1.0), _item("1-1-boom", 1.0)]
    transcribe_with_isolation(model, batch)  # type: ignore[arg-type]
    # One whole-batch attempt, then one call per item.
    assert model.batch_calls[0] == 2
    assert model.batch_calls[1:] == [1, 1]


def test_empty_batch_is_a_noop() -> None:
    model: FakeAsrModel = FakeAsrModel()
    outcome = transcribe_with_isolation(model, [])  # type: ignore[arg-type]
    assert not outcome.succeeded
    assert not outcome.failed
    assert not outcome.whole_batch_failed
    assert outcome.systematic_error is None
