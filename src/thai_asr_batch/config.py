"""Typed configuration loading and device resolution."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import (dataclass, field)
from pathlib import Path
from typing import (Any, Dict, List, Optional)

import yaml

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PathsConfig:
    audio_root: Path
    output_dir: Path
    checkpoint_dir: Path
    log_dir: Path
    scratch_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    hf_repo_id: str
    nemo_filename: str
    local_model_path: Optional[Path]
    target_lang: str
    strip_lang_tags: bool


@dataclass(frozen=True)
class DeviceConfig:
    prefer: List[str]
    cuda_index: int
    allow_mps: bool


@dataclass(frozen=True)
class BatchConfig:
    batch_size: int
    sort_buffer_size: int
    max_batch_duration_seconds: float


@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int
    mono: bool
    resample_on_the_fly: bool
    ffmpeg_binary: str
    ffmpeg_timeout_seconds: float


@dataclass(frozen=True)
class CheckpointConfig:
    enabled: bool
    flush_every_n: int
    flush_every_seconds: float
    max_cached_shards: int


@dataclass(frozen=True)
class RuntimeConfig:
    io_workers: int
    progress_every_n: int
    max_consecutive_batch_failures: int
    log_level: str


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    model: ModelConfig
    device: DeviceConfig
    batch: BatchConfig
    audio: AudioConfig
    checkpoint: CheckpointConfig
    runtime: RuntimeConfig
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def fingerprint(self) -> str:
        """Stable hash of the settings that must not change across a resume."""
        material: Dict[str, Any] = {
            "audio_root": str(self.paths.audio_root),
            "target_lang": self.model.target_lang,
            "strip_lang_tags": self.model.strip_lang_tags,
            "hf_repo_id": self.model.hf_repo_id,
        }
        blob: str = json.dumps(material, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if value is None:
            continue
        existing: Any = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _optional_path(value: Optional[str]) -> Optional[Path]:
    return Path(value).expanduser() if value else None


def load_config(
    path: Path,
    overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """Load YAML config, apply overrides, and return the typed tree."""
    with path.open("r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle) or {}

    if overrides:
        raw = _deep_merge(raw, overrides)

    paths_raw: Dict[str, Any] = raw.get("paths", {})
    model_raw: Dict[str, Any] = raw.get("model", {})
    device_raw: Dict[str, Any] = raw.get("device", {})
    batch_raw: Dict[str, Any] = raw.get("batch", {})
    audio_raw: Dict[str, Any] = raw.get("audio", {})
    ckpt_raw: Dict[str, Any] = raw.get("checkpoint", {})
    runtime_raw: Dict[str, Any] = raw.get("runtime", {})

    return AppConfig(
        paths=PathsConfig(
            audio_root=Path(paths_raw.get("audio_root", "./data/audio")).expanduser(),
            output_dir=Path(paths_raw.get("output_dir", "./data/predictions")).expanduser(),
            checkpoint_dir=Path(
                paths_raw.get("checkpoint_dir", "./data/checkpoints")
            ).expanduser(),
            log_dir=Path(paths_raw.get("log_dir", "./logs")).expanduser(),
            scratch_dir=Path(paths_raw.get("scratch_dir", "./data/scratch")).expanduser(),
        ),
        model=ModelConfig(
            hf_repo_id=model_raw.get(
                "hf_repo_id", "typhoon-ai/typhoon-asr-streaming-nemotron-0.6b"
            ),
            nemo_filename=model_raw.get(
                "nemo_filename", "typhoon-asr-streaming-nemotron-0.6b.nemo"
            ),
            local_model_path=_optional_path(model_raw.get("local_model_path")),
            target_lang=model_raw.get("target_lang", "th-TH"),
            strip_lang_tags=bool(model_raw.get("strip_lang_tags", True)),
        ),
        device=DeviceConfig(
            prefer=list(device_raw.get("prefer", ["cuda", "cpu"])),
            cuda_index=int(device_raw.get("cuda_index", 0)),
            allow_mps=bool(device_raw.get("allow_mps", False)),
        ),
        batch=BatchConfig(
            batch_size=int(batch_raw.get("batch_size", 16)),
            sort_buffer_size=int(batch_raw.get("sort_buffer_size", 2048)),
            max_batch_duration_seconds=float(
                batch_raw.get("max_batch_duration_seconds", 480.0)
            ),
        ),
        audio=AudioConfig(
            target_sample_rate=int(audio_raw.get("target_sample_rate", 16000)),
            mono=bool(audio_raw.get("mono", True)),
            resample_on_the_fly=bool(audio_raw.get("resample_on_the_fly", True)),
            ffmpeg_binary=audio_raw.get("ffmpeg_binary", "ffmpeg"),
            ffmpeg_timeout_seconds=float(audio_raw.get("ffmpeg_timeout_seconds", 60.0)),
        ),
        checkpoint=CheckpointConfig(
            enabled=bool(ckpt_raw.get("enabled", True)),
            flush_every_n=int(ckpt_raw.get("flush_every_n", 200)),
            flush_every_seconds=float(ckpt_raw.get("flush_every_seconds", 30.0)),
            max_cached_shards=int(ckpt_raw.get("max_cached_shards", 4)),
        ),
        runtime=RuntimeConfig(
            io_workers=int(runtime_raw.get("io_workers", 8)),
            progress_every_n=int(runtime_raw.get("progress_every_n", 500)),
            max_consecutive_batch_failures=int(
                runtime_raw.get("max_consecutive_batch_failures", 5)
            ),
            log_level=str(runtime_raw.get("log_level", "INFO")),
        ),
        raw=raw,
    )


def resolve_device(cfg: DeviceConfig) -> str:
    """Pick a torch device string by walking ``prefer`` in order.

    MPS is skipped unless explicitly opted in: NeMo on Apple MPS tends to fail
    outright for RNN-T rather than degrade gracefully, so a silent selection
    would turn a local smoke test into a confusing crash.
    """
    for candidate in cfg.prefer:
        name: str = candidate.lower()
        if name == "cuda" and _cuda_available():
            return f"cuda:{cfg.cuda_index}"
        if name == "mps":
            if not cfg.allow_mps:
                LOGGER.warning(
                    "device 'mps' requested but allow_mps is false; skipping "
                    "(NeMo RNN-T support on MPS is unreliable)"
                )
                continue
            if _mps_available():
                return "mps"
        if name == "cpu":
            return "cpu"
    LOGGER.warning("no preferred device available; falling back to cpu")
    return "cpu"


def _cuda_available() -> bool:
    try:
        import torch  # local import: torch is heavy and optional for dry runs
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    backend: Any = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())
