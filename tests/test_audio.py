"""Audio decoding and the Whisper encoder-window check.

No GPU and no transformers needed: load_samples uses soundfile/numpy, which are
ordinary locked dependencies.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from thai_asr_batch.audio import (WHISPER_WINDOW_SECONDS, AudioError,
                                  estimate_duration, exceeds_window,
                                  load_samples, probe_audio)
from thai_asr_batch.config import (AppConfig, AudioConfig)

from conftest import write_wav


def test_load_samples_returns_mono_float32(tmp_path: Path, app_config: AppConfig) -> None:
    path: Path = write_wav(tmp_path / "clip.wav", duration_seconds=0.25)
    samples: Any = load_samples(path, app_config.audio)

    assert samples.ndim == 1, "Whisper's processor wants a 1-D mono array"
    assert samples.dtype.name == "float32"
    # 0.25s at 16 kHz, allowing for header rounding.
    assert math.isclose(len(samples), 4000, rel_tol=0.02)


def test_load_samples_downmixes_stereo(tmp_path: Path, app_config: AppConfig) -> None:
    path: Path = write_wav(tmp_path / "stereo.wav", duration_seconds=0.1, channels=2)
    samples: Any = load_samples(path, app_config.audio)
    assert samples.ndim == 1


def test_load_samples_resamples_off_rate_audio(
    tmp_path: Path, app_config: AppConfig
) -> None:
    """A wrong sample rate must be corrected, not fed to the model as-is."""
    path: Path = write_wav(tmp_path / "8k.wav", duration_seconds=0.5, sample_rate=8000)
    samples: Any = load_samples(path, app_config.audio)
    # 0.5s at the 16 kHz target, regardless of the 8 kHz source.
    assert math.isclose(len(samples), 8000, rel_tol=0.05)


def test_load_samples_rejects_unreadable_files(tmp_path: Path, app_config: AppConfig) -> None:
    bogus: Path = tmp_path / "not-audio.wav"
    bogus.write_text("this is not a wav file", encoding="utf-8")
    with pytest.raises(AudioError):
        load_samples(bogus, app_config.audio)


def test_load_samples_defaults_to_16k_without_config(tmp_path: Path) -> None:
    path: Path = write_wav(tmp_path / "clip.wav", duration_seconds=0.1)
    samples: Any = load_samples(path, None)
    assert samples.ndim == 1


def test_exceeds_window_uses_the_configured_limit(app_config: AppConfig) -> None:
    assert not exceeds_window(29.9, app_config.audio)
    assert exceeds_window(30.1, app_config.audio)


def test_exceeds_window_boundary_is_not_exceeded(app_config: AppConfig) -> None:
    """Exactly 30s fits; only longer truncates."""
    assert not exceeds_window(app_config.audio.max_duration_seconds, app_config.audio)


def test_exceeds_window_falls_back_to_the_whisper_default() -> None:
    assert WHISPER_WINDOW_SECONDS == 30.0
    assert not exceeds_window(10.0, None)
    assert exceeds_window(45.0, None)


def test_exceeds_window_honours_an_override(app_config: AppConfig) -> None:
    from dataclasses import replace

    tighter: AudioConfig = replace(app_config.audio, max_duration_seconds=5.0)
    assert exceeds_window(6.0, tighter)
    assert not exceeds_window(4.0, tighter)


def test_estimate_duration_matches_the_written_clip(tmp_path: Path) -> None:
    path: Path = write_wav(tmp_path / "clip.wav", duration_seconds=0.5)
    assert math.isclose(estimate_duration(path), 0.5, rel_tol=0.02)


def test_long_clip_is_detectable_end_to_end(tmp_path: Path, app_config: AppConfig) -> None:
    """A >30s clip must be flagged by probe+exceeds_window, not silently pass."""
    path: Path = write_wav(tmp_path / "long.wav", duration_seconds=31.0)
    probe = probe_audio(path)
    assert probe.is_readable
    assert exceeds_window(probe.duration_seconds or 0.0, app_config.audio)
