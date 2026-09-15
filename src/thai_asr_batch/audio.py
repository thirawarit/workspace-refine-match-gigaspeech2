"""Audio probing, resample-on-the-fly, and sample loading.

GigaSpeech2 ships 16 kHz mono 16-bit PCM WAV, which already matches what the
model wants. So the conforming path is a header read only — no subprocess. At
10M clips, spawning ffmpeg per file would dominate total runtime.
"""

from __future__ import annotations

import logging
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Optional, Tuple)

from .config import AudioConfig

LOGGER: logging.Logger = logging.getLogger(__name__)

BYTES_PER_SAMPLE_ASSUMED: int = 2
WAV_HEADER_BYTES: int = 44

# Whisper encodes a fixed 30-second window; anything longer is truncated by the
# feature extractor rather than rejected, so it must be detected explicitly.
WHISPER_WINDOW_SECONDS: float = 30.0


class AudioError(RuntimeError):
    """Raised when a clip cannot be read or converted."""


@dataclass(frozen=True)
class AudioProbe:
    path: Path
    exists: bool
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None

    @property
    def is_readable(self) -> bool:
        return self.exists and self.error is None


def probe_audio(path: Path) -> AudioProbe:
    """Read WAV header fields without decoding the audio body."""
    if not path.exists():
        return AudioProbe(path=path, exists=False, error="file not found")

    try:
        with wave.open(str(path), "rb") as handle:
            frames: int = handle.getnframes()
            rate: int = handle.getframerate()
            channels: int = handle.getnchannels()
            duration: Optional[float] = (frames / rate) if rate else None
            return AudioProbe(
                path=path,
                exists=True,
                sample_rate=rate,
                channels=channels,
                duration_seconds=duration,
            )
    except (wave.Error, OSError, EOFError) as exc:
        return AudioProbe(path=path, exists=True, error=f"unreadable wav: {exc}")


def conforms(probe: AudioProbe, cfg: AudioConfig) -> bool:
    if not probe.is_readable:
        return False
    expected_channels: int = 1 if cfg.mono else (probe.channels or 1)
    return (
        probe.sample_rate == cfg.target_sample_rate
        and probe.channels == expected_channels
    )


def estimate_duration(path: Path, probe: Optional[AudioProbe] = None) -> float:
    """Best-effort duration for length bucketing only. Never raises.

    Bucketing tolerates approximation, so a bad header falls back to file-size
    arithmetic rather than failing the clip.
    """
    resolved: AudioProbe = probe if probe is not None else probe_audio(path)
    if resolved.duration_seconds is not None:
        return resolved.duration_seconds

    try:
        size: int = path.stat().st_size
    except OSError:
        return 0.0
    payload: int = max(0, size - WAV_HEADER_BYTES)
    return payload / float(16000 * BYTES_PER_SAMPLE_ASSUMED)


def ensure_conforming(
    path: Path,
    cfg: AudioConfig,
    scratch_dir: Path,
    probe: Optional[AudioProbe] = None,
) -> Tuple[Path, bool]:
    """Return a path the decoder can consume, plus whether a conversion happened.

    Fast path: conforming files are returned unchanged. Only a mismatch pays
    for an ffmpeg subprocess.
    """
    resolved: AudioProbe = probe if probe is not None else probe_audio(path)
    if not resolved.exists:
        raise AudioError(f"missing audio file: {path}")

    if not cfg.resample_on_the_fly:
        if resolved.error is not None:
            raise AudioError(f"{path}: {resolved.error}")
        return path, False

    if conforms(resolved, cfg):
        return path, False

    LOGGER.debug(
        "%s is %s Hz / %s ch (want %s Hz / %s ch); converting",
        path, resolved.sample_rate, resolved.channels,
        cfg.target_sample_rate, 1 if cfg.mono else resolved.channels,
    )
    return _convert_with_ffmpeg(path, cfg, scratch_dir), True


def exceeds_window(duration_seconds: float, cfg: Optional[AudioConfig] = None) -> bool:
    """True when a clip is longer than the model's fixed encoder window.

    Whisper silently keeps only the first window, so a long clip yields partial
    text that looks like a bad transcription rather than a truncation. Callers
    flag these instead of letting them pass unnoticed.
    """
    limit: float = (
        cfg.max_duration_seconds if cfg is not None else WHISPER_WINDOW_SECONDS
    )
    return duration_seconds > limit


def load_samples(path: Path, cfg: Optional[AudioConfig] = None) -> Any:
    """Decode an audio file to a mono float32 array at the target sample rate.

    Whisper's processor consumes arrays rather than paths. soundfile and numpy
    are already project dependencies, so this needs no extra decode library.
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency is locked
        raise AudioError(f"soundfile/numpy required to decode audio: {exc}") from exc

    target_rate: int = cfg.target_sample_rate if cfg is not None else 16000

    try:
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - soundfile raises several types
        raise AudioError(f"could not decode {path}: {exc}") from exc

    # Downmix to mono; GigaSpeech2 is already mono, so this is usually a no-op.
    mono: Any = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]

    if sample_rate != target_rate:
        # ensure_conforming normally resamples via ffmpeg upstream; this is a
        # last-resort linear fallback so an unexpected rate degrades rather
        # than feeding the model mis-rated audio.
        LOGGER.warning(
            "%s is %d Hz, expected %d; resampling in-process",
            path, sample_rate, target_rate,
        )
        duration: float = len(mono) / float(sample_rate)
        target_len: int = max(1, int(round(duration * target_rate)))
        mono = np.interp(
            np.linspace(0.0, len(mono), num=target_len, endpoint=False),
            np.arange(len(mono)),
            mono,
        ).astype("float32")

    return mono


def _convert_with_ffmpeg(path: Path, cfg: AudioConfig, scratch_dir: Path) -> Path:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    target: Path = scratch_dir / f"{path.stem}-{cfg.target_sample_rate}{path.suffix or '.wav'}"

    command: list[str] = [
        cfg.ffmpeg_binary,
        "-nostdin", "-loglevel", "error", "-y",
        "-i", str(path),
        "-ar", str(cfg.target_sample_rate),
        "-ac", "1" if cfg.mono else "2",
        "-c:a", "pcm_s16le",
        str(target),
    ]

    try:
        result: subprocess.CompletedProcess[bytes] = subprocess.run(
            command,
            capture_output=True,
            timeout=cfg.ffmpeg_timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AudioError(f"ffmpeg binary {cfg.ffmpeg_binary!r} not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"ffmpeg timed out after {cfg.ffmpeg_timeout_seconds}s on {path}") from exc

    if result.returncode != 0:
        detail: str = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioError(f"ffmpeg failed on {path}: {detail}")

    return target
