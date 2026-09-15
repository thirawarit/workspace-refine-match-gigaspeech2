"""Config loading, device policy, and the CLI surface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pytest

from thai_asr_batch.cli import get_parser
from thai_asr_batch.config import (AppConfig, DeviceConfig, load_config,
                                   resolve_device)

DEFAULT_CONFIG: Path = Path("configs/default.yaml")


def test_default_config_loads() -> None:
    cfg: AppConfig = load_config(DEFAULT_CONFIG)
    # Whisper takes ISO-639-1 ("th"), not a BCP-47 tag ("th-TH").
    assert cfg.model.target_lang == "th"
    assert cfg.model.task == "transcribe"
    assert cfg.model.hf_repo_id == "typhoon-ai/typhoon-whisper-medium"
    assert cfg.batch.batch_size == 16
    assert cfg.audio.target_sample_rate == 16000


def test_whisper_window_is_configured() -> None:
    """The 30s encoder window must be explicit, not implied."""
    cfg: AppConfig = load_config(DEFAULT_CONFIG)
    assert cfg.audio.max_duration_seconds == 30.0
    assert cfg.model.max_new_tokens == 440
    assert cfg.model.dtype in {"bfloat16", "float16", "float32"}


def test_cuda_index_is_pinned_to_zero() -> None:
    """GPU index 1 is fully occupied on the VPS."""
    cfg: AppConfig = load_config(DEFAULT_CONFIG)
    assert cfg.device.cuda_index == 0


def test_overrides_are_applied() -> None:
    cfg: AppConfig = load_config(DEFAULT_CONFIG, {"batch": {"batch_size": 99}})
    assert cfg.batch.batch_size == 99


def test_overrides_merge_rather_than_replace() -> None:
    cfg: AppConfig = load_config(DEFAULT_CONFIG, {"batch": {"batch_size": 99}})
    # sort_buffer_size survives an override that only names batch_size.
    assert cfg.batch.sort_buffer_size == 2048


def test_fingerprint_changes_with_audio_root() -> None:
    base: AppConfig = load_config(DEFAULT_CONFIG)
    moved: AppConfig = load_config(DEFAULT_CONFIG, {"paths": {"audio_root": "/elsewhere"}})
    assert base.fingerprint() != moved.fingerprint()


def test_fingerprint_is_stable() -> None:
    assert load_config(DEFAULT_CONFIG).fingerprint() == load_config(DEFAULT_CONFIG).fingerprint()


def test_mps_is_never_chosen_implicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """NeMo RNN-T on MPS tends to fail outright, so it needs an explicit opt-in."""
    monkeypatch.setattr("thai_asr_batch.config._cuda_available", lambda: False)
    monkeypatch.setattr("thai_asr_batch.config._mps_available", lambda: True)
    cfg: DeviceConfig = DeviceConfig(prefer=["cuda", "mps", "cpu"], cuda_index=0,
                                     allow_mps=False)
    assert resolve_device(cfg) == "cpu"


def test_mps_is_used_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("thai_asr_batch.config._cuda_available", lambda: False)
    monkeypatch.setattr("thai_asr_batch.config._mps_available", lambda: True)
    cfg: DeviceConfig = DeviceConfig(prefer=["mps", "cpu"], cuda_index=0, allow_mps=True)
    assert resolve_device(cfg) == "mps"


def test_cuda_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("thai_asr_batch.config._cuda_available", lambda: False)
    cfg: DeviceConfig = DeviceConfig(prefer=["cuda", "cpu"], cuda_index=0, allow_mps=False)
    assert resolve_device(cfg) == "cpu"


def test_cuda_uses_the_configured_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("thai_asr_batch.config._cuda_available", lambda: True)
    cfg: DeviceConfig = DeviceConfig(prefer=["cuda"], cuda_index=0, allow_mps=False)
    assert resolve_device(cfg) == "cuda:0"


def test_local_cpu_profile_disables_gpu() -> None:
    cfg: AppConfig = load_config(Path("configs/local_cpu.yaml"))
    assert cfg.device.prefer == ["cpu"]
    assert cfg.device.allow_mps is False


@pytest.mark.parametrize(
    "command",
    ["transcribe", "combine", "run", "validate", "status"],
)
def test_parser_accepts_each_subcommand(command: str) -> None:
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args([command, "--input", "x.tsv"])
    assert args.command == command


def test_parser_exposes_the_documented_flags() -> None:
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args([
        "transcribe", "--input", "x.tsv", "--limit", "10", "--dry-run",
        "--no-resume", "--batch-size", "4", "--device", "cpu",
    ])
    assert args.limit == 10
    assert args.dry_run is True
    assert args.no_resume is True
    assert args.batch_size == 4
    assert args.device == "cpu"


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        get_parser().parse_args([])


def test_parser_requires_input() -> None:
    with pytest.raises(SystemExit):
        get_parser().parse_args(["transcribe"])
