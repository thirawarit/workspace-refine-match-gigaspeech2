"""Shared fixtures. No GPU, no NeMo, no binary fixtures in git."""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import (List, Sequence)

import pytest

from thai_asr_batch.config import (AppConfig, load_config)

SAMPLE_RATE: int = 16000

# Thai text exercising the NFC trap: สระอำ has two encodings that render
# identically but compare unequal without normalization.
THAI_COMPOSED: str = "คำ"            # คำ  (U+0E33)
THAI_DECOMPOSED: str = "คํา"    # ค + นิคหิต + สระอา


def write_wav(
    path: Path,
    duration_seconds: float = 0.1,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
) -> Path:
    """Write a real, tiny, valid WAV via the stdlib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames: int = max(1, int(duration_seconds * sample_rate))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack("<h", 0) * frames * channels)
    return path


@pytest.fixture()
def audio_root(tmp_path: Path) -> Path:
    """Audio tree matching the <root>/train/<idx0>/<idx1>/<sid>.wav rule."""
    root: Path = tmp_path / "audio"
    for segment_id, duration in (
        ("100-100000-0", 0.10),
        ("100-100000-1", 0.20),
        ("100-100000-2", 0.05),
        ("148-148801-10", 0.30),
        ("148-148801-11", 0.15),
    ):
        parts: List[str] = segment_id.split("-")
        write_wav(root / "train" / parts[0] / parts[1] / f"{segment_id}.wav", duration)
    return root


@pytest.fixture()
def sample_tsv(tmp_path: Path) -> Path:
    """Headerless two-column TSV in the shape the user supplied."""
    path: Path = tmp_path / "sample.tsv"
    rows: List[str] = [
        "100-100000-0\tท่านผู้ชมครับเรื่องของ COVID-19 วันนี้ไทยพบผู้ป่วยเพิ่มหนึ่งคนนะครับ",
        "100-100000-1\tเป็นผู้หญิงอายุยี่สิบสองอาชีพดูแลนักท่องเที่ยว",
        "100-100000-2\tสัมผัสกับกลุ่มผู้ที่มีความเสี่ยงสูง",
        "148-148801-10\tดำเนินการจัดการเคลียร์พื้นที่ดังกล่าว",
        "148-148801-11\tให้ทันภายใน24พฤศจิกายนนี้",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def sample_jsonl(tmp_path: Path, audio_root: Path) -> Path:
    """NeMo-manifest JSONL; segment ids derive from the audio_filepath stems."""
    import json

    path: Path = tmp_path / "sample.jsonl"
    lines: List[str] = []
    for segment_id, text in (
        ("100-100000-0", "ท่านผู้ชมครับ"),
        ("148-148801-11", "ให้ทันภายใน24พฤศจิกายนนี้"),
    ):
        parts: List[str] = segment_id.split("-")
        audio: Path = audio_root / "train" / parts[0] / parts[1] / f"{segment_id}.wav"
        lines.append(json.dumps(
            {"audio_filepath": str(audio), "text": text, "duration": 0.1},
            ensure_ascii=False,
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def app_config(tmp_path: Path, audio_root: Path) -> AppConfig:
    config_path: Path = Path("configs/default.yaml")
    return load_config(config_path, {
        "paths": {
            "audio_root": str(audio_root),
            "output_dir": str(tmp_path / "predictions"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
            "scratch_dir": str(tmp_path / "scratch"),
        },
        "batch": {"batch_size": 2, "sort_buffer_size": 8},
        "checkpoint": {"flush_every_n": 1, "flush_every_seconds": 0.0},
        "runtime": {"io_workers": 2, "progress_every_n": 100},
        "device": {"prefer": ["cpu"]},
    })


class FakeAsrModel:
    """Stand-in for AsrModelWrapper. Deterministic, and fails on 'boom' ids."""

    def __init__(self, fail_substring: str = "boom") -> None:
        self.fail_substring: str = fail_substring
        self.batch_calls: List[int] = []
        self._loaded: bool = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def transcribe_batch(self, audio_paths: Sequence[Path]) -> List[str]:
        self.batch_calls.append(len(audio_paths))
        for path in audio_paths:
            if self.fail_substring in path.stem:
                raise RuntimeError(f"synthetic failure on {path.stem}")
        return [f"pred:{path.stem}" for path in audio_paths]


@pytest.fixture()
def fake_model() -> FakeAsrModel:
    return FakeAsrModel()
