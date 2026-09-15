"""Orchestration: resume, limit, failure isolation, dead-context abort.

Runs entirely without NeMo or a GPU — guaranteed by the deferred import in
model.load(), which these tests never trigger.
"""

from __future__ import annotations

from pathlib import Path
from typing import (Dict, List)

import pytest

from thai_asr_batch.batching import DeadContextError
from thai_asr_batch.config import AppConfig
from thai_asr_batch.inference import (InferenceStats, run_inference)
from thai_asr_batch.io_formats import predicted_output_path

from conftest import (FakeAsrModel, write_wav)


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch: pytest.MonkeyPatch) -> FakeAsrModel:
    """Swap the NeMo wrapper for the fake everywhere inference constructs one."""
    fake: FakeAsrModel = FakeAsrModel()
    monkeypatch.setattr(
        "thai_asr_batch.inference.AsrModelWrapper",
        lambda cfg, device: fake,
    )
    return fake


def _predictions(app_config: AppConfig, source: Path) -> Dict[str, str]:
    path: Path = predicted_output_path(source, app_config.paths.output_dir)
    if not path.exists():
        return {}
    result: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sid, _, text = line.partition("\t")
        result[sid] = text
    return result


def test_transcribes_every_row(app_config: AppConfig, sample_tsv: Path) -> None:
    stats: InferenceStats = run_inference(sample_tsv, app_config)
    assert stats.succeeded == 5
    assert stats.failed == 0
    assert len(_predictions(app_config, sample_tsv)) == 5


def test_output_lands_at_predicted_name(app_config: AppConfig, sample_tsv: Path) -> None:
    run_inference(sample_tsv, app_config)
    expected: Path = app_config.paths.output_dir / "predicted-sample.tsv"
    assert expected.exists()


def test_limit_stops_intake(app_config: AppConfig, sample_tsv: Path) -> None:
    stats: InferenceStats = run_inference(sample_tsv, app_config, limit=2)
    assert stats.succeeded == 2


def test_resume_skips_completed_rows(app_config: AppConfig, sample_tsv: Path) -> None:
    first: InferenceStats = run_inference(sample_tsv, app_config, limit=2)
    assert first.succeeded == 2

    second: InferenceStats = run_inference(sample_tsv, app_config)
    assert second.skipped_resume == 2
    assert second.succeeded == 3
    assert len(_predictions(app_config, sample_tsv)) == 5


def test_rerun_after_completion_does_nothing(
    app_config: AppConfig, sample_tsv: Path
) -> None:
    run_inference(sample_tsv, app_config)
    again: InferenceStats = run_inference(sample_tsv, app_config)
    assert again.succeeded == 0
    assert again.skipped_resume == 5


def test_no_resume_starts_over(app_config: AppConfig, sample_tsv: Path) -> None:
    run_inference(sample_tsv, app_config)
    fresh: InferenceStats = run_inference(sample_tsv, app_config, resume=False)
    assert fresh.skipped_resume == 0
    assert fresh.succeeded == 5


def test_missing_audio_is_skipped_not_fatal(
    app_config: AppConfig, tmp_path: Path
) -> None:
    source: Path = tmp_path / "partial.tsv"
    source.write_text(
        "100-100000-0\tpresent\n999-999999-9\tabsent\n", encoding="utf-8"
    )
    stats: InferenceStats = run_inference(source, app_config)
    assert stats.succeeded == 1
    assert stats.skipped_missing_audio == 1


def test_poison_clip_gets_empty_prediction(
    app_config: AppConfig, tmp_path: Path, audio_root: Path
) -> None:
    """A failing clip must yield an empty pred_text, never abort the run."""
    write_wav(audio_root / "train" / "500" / "500001" / "500-500001-boom.wav", 0.1)
    source: Path = tmp_path / "withboom.tsv"
    source.write_text(
        "100-100000-0\tfine\n500-500001-boom\tbad\n100-100000-1\talso fine\n",
        encoding="utf-8",
    )

    stats: InferenceStats = run_inference(source, app_config)
    assert stats.succeeded == 2
    assert stats.failed == 1
    assert _predictions(app_config, source)["500-500001-boom"] == ""


def test_dead_context_aborts(
    app_config: AppConfig, tmp_path: Path, audio_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consecutive whole-batch failures must stop the run, not write empties."""
    from dataclasses import replace

    cfg: AppConfig = replace(
        app_config, runtime=replace(app_config.runtime, max_consecutive_batch_failures=2)
    )

    ids: List[str] = []
    for index in range(8):
        segment_id: str = f"600-600001-boom{index}"
        ids.append(segment_id)
        write_wav(audio_root / "train" / "600" / "600001" / f"{segment_id}.wav", 0.1)

    source: Path = tmp_path / "allbad.tsv"
    source.write_text("\n".join(f"{sid}\tx" for sid in ids) + "\n", encoding="utf-8")

    with pytest.raises(DeadContextError):
        run_inference(source, cfg)


def test_dry_run_never_loads_the_model(
    app_config: AppConfig, sample_tsv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry run must not construct the model")

    monkeypatch.setattr("thai_asr_batch.inference.AsrModelWrapper", explode)
    stats: InferenceStats = run_inference(sample_tsv, app_config, dry_run=True)
    assert stats.succeeded == 5


def test_dry_run_reports_missing_audio(app_config: AppConfig, tmp_path: Path) -> None:
    source: Path = tmp_path / "missing.tsv"
    source.write_text("999-999999-9\tabsent\n", encoding="utf-8")
    stats: InferenceStats = run_inference(source, app_config, dry_run=True)
    assert stats.skipped_missing_audio == 1


def test_jsonl_input_round_trips(app_config: AppConfig, sample_jsonl: Path) -> None:
    stats: InferenceStats = run_inference(sample_jsonl, app_config)
    assert stats.succeeded == 2
    output: Path = predicted_output_path(sample_jsonl, app_config.paths.output_dir)
    assert output.suffix == ".jsonl"
