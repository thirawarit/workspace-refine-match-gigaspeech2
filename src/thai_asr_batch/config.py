"""Typed configuration loading and device resolution."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import (dataclass, field)
from pathlib import Path
from typing import (Any, Dict, List, Optional, Tuple)

import yaml

LOGGER: logging.Logger = logging.getLogger(__name__)


class DeviceConfigError(RuntimeError):
    """Raised when a requested device cannot be honoured.

    Always raised for an unparseable spec, and — when the request was explicit
    (a CLI flag rather than a YAML preference list) — for a CUDA device that is
    unavailable or out of range. A silent CPU fallback on a ten-day corpus is a
    far worse outcome than an abort.
    """


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
    local_model_path: Optional[Path]
    target_lang: str
    task: str
    dtype: str
    max_new_tokens: int
    sampling_rate: int


@dataclass(frozen=True)
class DeviceConfig:
    # Entries are 'cpu', 'mps', 'cuda' or 'cuda:N'; see resolve_device.
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
    max_duration_seconds: float


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
            "task": self.model.task,
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
            hf_repo_id=model_raw.get("hf_repo_id", "typhoon-ai/typhoon-whisper-medium"),
            local_model_path=_optional_path(model_raw.get("local_model_path")),
            # Whisper uses ISO-639-1 ("th"), not a BCP-47 tag ("th-TH").
            target_lang=model_raw.get("target_lang", "th"),
            task=model_raw.get("task", "transcribe"),
            dtype=model_raw.get("dtype", "bfloat16"),
            max_new_tokens=int(model_raw.get("max_new_tokens", 440)),
            sampling_rate=int(model_raw.get("sampling_rate", 16000)),
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
            max_duration_seconds=float(audio_raw.get("max_duration_seconds", 30.0)),
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


def resolve_device(cfg: DeviceConfig, explicit: bool = False) -> str:
    """Pick a torch device string by walking ``prefer`` in order.

    Each entry is ``cpu``, ``mps``, ``cuda`` or ``cuda:N``. A bare ``cuda`` uses
    ``cfg.cuda_index``; an explicit ``cuda:N`` overrides it.

    MPS is skipped unless explicitly opted in: Whisper generation on Apple MPS is
    slow and historically flaky, so a silent selection would turn a local smoke
    test into a confusing hang.

    ``explicit`` marks a device the user named on the command line. Such a
    request never degrades to CPU — it raises :class:`DeviceConfigError` instead,
    because a ten-day run that silently crawls on CPU is the failure this whole
    function exists to prevent. A YAML ``prefer`` list keeps the soft fallback.
    """
    for candidate in cfg.prefer:
        kind: str
        index: Optional[int]
        kind, index = _parse_device_spec(candidate)

        if kind == "cuda":
            resolved: int = cfg.cuda_index if index is None else index
            if _cuda_available():
                _check_cuda_index(resolved, candidate)
                return f"cuda:{resolved}"
            if explicit:
                raise DeviceConfigError(
                    f"device {candidate!r} was requested explicitly but CUDA is "
                    "unavailable. Check `nvidia-smi`, or drop the flag to fall "
                    "back to CPU."
                )
            LOGGER.warning("cuda requested but unavailable; trying the next preference")
            continue

        if kind == "mps":
            if not cfg.allow_mps:
                if explicit:
                    raise DeviceConfigError(
                        "device 'mps' was requested explicitly but allow_mps is "
                        "false. Set device.allow_mps: true to opt in."
                    )
                LOGGER.warning(
                    "device 'mps' requested but allow_mps is false; skipping "
                    "(Whisper generation on MPS is slow and historically flaky)"
                )
                continue
            if _mps_available():
                return "mps"
            if explicit:
                raise DeviceConfigError(
                    "device 'mps' was requested explicitly but MPS is unavailable."
                )
            continue

        if kind == "cpu":
            return "cpu"

    if explicit:
        raise DeviceConfigError(
            f"none of the requested devices are available: {list(cfg.prefer)!r}"
        )
    LOGGER.warning("no preferred device available; falling back to cpu")
    return "cpu"


def _parse_device_spec(spec: str) -> Tuple[str, Optional[int]]:
    """Split a device spec into its kind and optional index.

    Accepts ``cpu``, ``mps``, ``cuda`` and ``cuda:N`` (case-insensitive).
    Anything else raises: before this existed, an unmatched spec fell through
    every branch and landed on the CPU fallback, so ``--device cuda:1`` — and
    even ``--device cuda:0`` — silently ran the whole corpus on CPU.
    """
    name: str = spec.strip().lower()
    if name in {"cpu", "mps", "cuda"}:
        return (name, None)

    head, sep, tail = name.partition(":")
    if head == "cuda" and sep and tail.isdigit():
        return ("cuda", int(tail))

    raise DeviceConfigError(
        f"unsupported device spec {spec!r}; expected one of "
        "'cpu', 'mps', 'cuda' or 'cuda:N'"
    )


def _check_cuda_index(index: int, spec: str) -> None:
    """Validate a CUDA index against the visible device count.

    Caught here with a legible message rather than deep inside ``.to(device)``.
    ``CUDA_VISIBLE_DEVICES`` renumbers devices — under a mask of ``1`` the only
    visible GPU is ``cuda:0`` — so an index past the count usually means the two
    mechanisms were combined by mistake, and the message says so.
    """
    count: int = _cuda_device_count()
    if count and index >= count:
        mask: Optional[str] = os.environ.get("CUDA_VISIBLE_DEVICES")
        hint: str = ""
        if mask:
            hint = (
                f" CUDA_VISIBLE_DEVICES={mask!r} is set, which renumbers devices: "
                "the visible GPUs are always 0..N-1 regardless of their physical "
                "index, so combining it with a cuda index is usually a mistake."
            )
        raise DeviceConfigError(
            f"device {spec!r} requested but only {count} CUDA device(s) are "
            f"visible (valid indices 0..{count - 1}).{hint}"
        )


def _cuda_device_count() -> int:
    """Visible CUDA device count, or 0 when torch cannot report one."""
    try:
        import torch
    except (ImportError, OSError):
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 - a broken driver must not mask the real error
        return 0


def _cuda_available() -> bool:
    try:
        import torch  # local import: torch is heavy and optional for dry runs
    except ImportError:
        return False
    except OSError as exc:
        # torch is installed but a native library will not load, e.g.
        # "libcudnn.so.9: cannot open shared object file". Degrade to CPU rather
        # than crash, so --dry-run and validate still work on a broken box.
        LOGGER.warning(
            "torch is installed but failed to load a shared library (%s); "
            "treating CUDA as unavailable. The nvidia-* wheels may be partially "
            "extracted or missing from LD_LIBRARY_PATH — "
            "./setup_and_run.sh --setup-only repairs both.",
            exc,
        )
        return False
    return bool(torch.cuda.is_available())


def _mps_available() -> bool:
    try:
        import torch
    except (ImportError, OSError):
        return False
    backend: Any = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())
