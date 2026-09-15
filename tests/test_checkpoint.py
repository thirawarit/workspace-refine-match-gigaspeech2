"""Resume state: sharding, crash recovery, LRU eviction, manifest guarding.

Highest-value test module — this logic is the most likely to be subtly wrong
and the most expensive to get wrong on a multi-day run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Set

import pytest

from thai_asr_batch.checkpoint import (CheckpointMismatchError, CheckpointStore)
from thai_asr_batch.config import AppConfig


def _store(cfg: AppConfig, input_path: Path, **kwargs: object) -> CheckpointStore:
    return CheckpointStore(
        root=cfg.paths.checkpoint_dir,
        input_path=input_path,
        cfg=cfg.checkpoint,
        config_fingerprint=cfg.fingerprint(),
        input_format="tsv",
        **kwargs,  # type: ignore[arg-type]
    )


def test_marks_and_reloads_across_restart(app_config: AppConfig, sample_tsv: Path) -> None:
    store: CheckpointStore = _store(app_config, sample_tsv)
    store.open()
    store.mark_done("100-100000-0")
    store.mark_done("148-148801-11")
    store.close()

    reopened: CheckpointStore = _store(app_config, sample_tsv)
    reopened.open()
    assert reopened.is_done("100-100000-0")
    assert reopened.is_done("148-148801-11")
    assert not reopened.is_done("100-100000-2")
    reopened.close()


def test_state_is_sharded_by_first_component(
    app_config: AppConfig, sample_tsv: Path
) -> None:
    store: CheckpointStore = _store(app_config, sample_tsv)
    store.open()
    store.mark_done("100-100000-0")
    store.mark_done("148-148801-11")
    store.close()

    shards: Set[str] = {p.name for p in store.dir.glob("shard-*.done")}
    assert shards == {"shard-100.done", "shard-148.done"}


def test_recovers_from_a_truncated_shard(app_config: AppConfig, sample_tsv: Path) -> None:
    """Simulate kill -9 mid-append: the partial id must be dropped."""
    store: CheckpointStore = _store(app_config, sample_tsv)
    store.open()
    store.mark_done("100-100000-0")
    store.close()

    shard: Path = store.dir / "shard-100.done"
    with shard.open("a", encoding="utf-8") as handle:
        handle.write("100-10000")  # no trailing newline

    reopened: CheckpointStore = _store(app_config, sample_tsv)
    reopened.open()
    assert reopened.is_done("100-100000-0")
    assert not reopened.is_done("100-10000")
    assert shard.read_text(encoding="utf-8") == "100-100000-0\n"
    reopened.close()


def test_drops_malformed_lines(app_config: AppConfig, sample_tsv: Path) -> None:
    store: CheckpointStore = _store(app_config, sample_tsv)
    store.open()
    store.close()
    shard: Path = store.dir / "shard-100.done"
    shard.write_text("100-100000-0\ngarbage\n\n100-100000-1\n", encoding="utf-8")

    reopened: CheckpointStore = _store(app_config, sample_tsv)
    reopened.open()
    assert reopened.is_done("100-100000-0")
    assert reopened.is_done("100-100000-1")
    reopened.close()


def test_manifest_mismatch_is_refused(app_config: AppConfig, sample_tsv: Path) -> None:
    """Resuming against a changed input would interleave two corpora."""
    store: CheckpointStore = _store(app_config, sample_tsv)
    store.open()
    store.mark_done("100-100000-0")
    store.close()

    sample_tsv.write_text("100-100000-0\tdifferent content entirely\n", encoding="utf-8")

    with pytest.raises(CheckpointMismatchError):
        _store(app_config, sample_tsv).open()


def test_force_resume_overrides_mismatch(app_config: AppConfig, sample_tsv: Path) -> None:
    store: CheckpointStore = _store(app_config, sample_tsv)
    store.open()
    store.close()
    sample_tsv.write_text("100-100000-0\tchanged\n", encoding="utf-8")

    forced: CheckpointStore = _store(app_config, sample_tsv)
    forced.open(force=True)
    manifest = json.loads((forced.dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_size"] == sample_tsv.stat().st_size
    forced.close()


def test_lru_eviction_bounds_memory(app_config: AppConfig, sample_tsv: Path) -> None:
    store: CheckpointStore = _store(app_config, sample_tsv)
    store.open()
    for shard in ("100", "148", "200", "300", "400"):
        store.load_completed_for_shard(shard)
    assert len(store._cache) <= max(1, app_config.checkpoint.max_cached_shards)
    store.close()


def test_eviction_does_not_lose_durable_state(
    app_config: AppConfig, sample_tsv: Path
) -> None:
    store: CheckpointStore = _store(app_config, sample_tsv)
    store.open()
    store.mark_done("100-100000-0")
    store.flush()
    for shard in ("148", "200", "300", "400", "500"):
        store.load_completed_for_shard(shard)
    assert store.is_done("100-100000-0")
    store.close()


def test_count_completed(app_config: AppConfig, sample_tsv: Path) -> None:
    store: CheckpointStore = _store(app_config, sample_tsv)
    store.open()
    store.mark_done("100-100000-0")
    store.mark_done("100-100000-1")
    store.mark_done("148-148801-11")
    store.close()
    assert _store(app_config, sample_tsv).count_completed() == 3


def test_disabled_checkpoint_never_reports_done(
    app_config: AppConfig, sample_tsv: Path
) -> None:
    from dataclasses import replace

    disabled: AppConfig = replace(
        app_config, checkpoint=replace(app_config.checkpoint, enabled=False)
    )
    store: CheckpointStore = _store(disabled, sample_tsv)
    store.open()
    store.mark_done("100-100000-0")
    assert not store.is_done("100-100000-0")
    store.close()
