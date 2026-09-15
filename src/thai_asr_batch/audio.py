"""Audio probing and resample-on-the-fly.

GigaSpeech2 ships 16 kHz mono 16-bit PCM WAV, which already matches what NeMo
wants. So the conforming path is a header read only — no subprocess. At 10M
clips, spawning ffmpeg per file would dominate total runtime.
"""

from __future__ import annotations

import logging
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import (Optional, Tuple)

from .config import AudioConfig

LOGGER: logging.Logger = logging.getLogger(__name__)

BYTES_PER_SAMPLE_ASSUMED: int = 2
WAV_HEADER_BYTES: int = 44


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
    """Return a path NeMo can consume, plus whether a conversion happened.

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
