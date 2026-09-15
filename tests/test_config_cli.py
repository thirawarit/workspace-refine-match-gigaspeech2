"""Config loading, device policy, and the CLI surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from thai_asr_batch.cli import (_overrides_from_args, get_parser)
from thai_asr_batch.config import (AppConfig, DeviceConfig, DeviceConfigError,
                                   load_config, resolve_device)

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
    monkeypatch.setattr("thai_asr_batch.config._cuda_device_count", lambda: 4)
    cfg: DeviceConfig = DeviceConfig(prefer=["cuda"], cuda_index=0, allow_mps=False)
    assert resolve_device(cfg) == "cuda:0"


def test_indexed_cuda_spec_selects_that_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: 'cuda:N' matched no branch and silently resolved to cpu.

    Because --device maps to prefer=[value], `--device cuda:1` ran a ten-day
    corpus on CPU. Even 'cuda:0' fell through.
    """
    monkeypatch.setattr("thai_asr_batch.config._cuda_available", lambda: True)
    monkeypatch.setattr("thai_asr_batch.config._cuda_device_count", lambda: 4)
    for spec, expected in (("cuda:1", "cuda:1"), ("cuda:0", "cuda:0"),
                           ("CUDA:2", "cuda:2")):
        cfg: DeviceConfig = DeviceConfig(prefer=[spec], cuda_index=0, allow_mps=False)
        assert resolve_device(cfg) == expected


def test_indexed_spec_overrides_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("thai_asr_batch.config._cuda_available", lambda: True)
    monkeypatch.setattr("thai_asr_batch.config._cuda_device_count", lambda: 4)
    cfg: DeviceConfig = DeviceConfig(prefer=["cuda:3"], cuda_index=0, allow_mps=False)
    assert resolve_device(cfg) == "cuda:3"


def test_out_of_range_index_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caught here with a clear message, not deep inside .to(device)."""
    monkeypatch.setattr("thai_asr_batch.config._cuda_available", lambda: True)
    monkeypatch.setattr("thai_asr_batch.config._cuda_device_count", lambda: 2)
    cfg: DeviceConfig = DeviceConfig(prefer=["cuda:7"], cuda_index=0, allow_mps=False)
    with pytest.raises(DeviceConfigError, match="only 2 CUDA device"):
        resolve_device(cfg)


def test_unparseable_spec_always_raises() -> None:
    """A typo is never a reason to spend ten days on CPU."""
    cfg: DeviceConfig = DeviceConfig(prefer=["gpu"], cuda_index=0, allow_mps=False)
    with pytest.raises(DeviceConfigError, match="unsupported device spec"):
        resolve_device(cfg)


def test_explicit_cuda_request_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("thai_asr_batch.config._cuda_available", lambda: False)
    cfg: DeviceConfig = DeviceConfig(prefer=["cuda:1"], cuda_index=0, allow_mps=False)
    with pytest.raises(DeviceConfigError, match="CUDA is unavailable"):
        resolve_device(cfg, explicit=True)


def test_implicit_preference_still_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A YAML prefer list keeps the soft fallback; only CLI requests are hard."""
    monkeypatch.setattr("thai_asr_batch.config._cuda_available", lambda: False)
    cfg: DeviceConfig = DeviceConfig(prefer=["cuda", "cpu"], cuda_index=0, allow_mps=False)
    assert resolve_device(cfg, explicit=False) == "cpu"


def test_visible_devices_mask_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    """Masking renumbers devices, so mask=1 + index=1 is always wrong."""
    monkeypatch.setattr("thai_asr_batch.config._cuda_available", lambda: True)
    monkeypatch.setattr("thai_asr_batch.config._cuda_device_count", lambda: 1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    cfg: DeviceConfig = DeviceConfig(prefer=["cuda:1"], cuda_index=0, allow_mps=False)
    with pytest.raises(DeviceConfigError, match="renumbers devices"):
        resolve_device(cfg)


def test_cuda_index_flag_merges_with_prefer_list() -> None:
    """--cuda-index alone must not wipe the YAML prefer list."""
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args(
        ["transcribe", "--input", "x.tsv", "--cuda-index", "2"]
    )
    overrides = _overrides_from_args(args)
    assert overrides["device"] == {"cuda_index": 2}
    assert "prefer" not in overrides["device"]

    cfg: AppConfig = load_config(DEFAULT_CONFIG, overrides)
    assert cfg.device.cuda_index == 2
    assert cfg.device.prefer == ["cuda", "cpu"]


def test_device_flag_is_available_on_validate() -> None:
    """validate reports the resolved device, so it needs the flags too."""
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args(
        ["validate", "--input", "x.tsv", "--device", "cuda:1"]
    )
    assert args.device == "cuda:1"


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
