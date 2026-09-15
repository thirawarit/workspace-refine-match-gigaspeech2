"""The internal normal form shared by every pipeline stage, and the id/path rules."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import (List, Literal, Optional)

SEGMENT_ID_PARTS: int = 3
AUDIO_SUFFIX: str = ".wav"
AUDIO_SUBDIR: str = "train"

# NFKC rather than NFC: see normalize_text for why Thai สระอำ requires it.
NORMAL_FORM: Literal["NFKC"] = "NFKC"


class MalformedSegmentIdError(ValueError):
    """Raised when a segment id cannot be decomposed into an audio path."""


@dataclass(frozen=True)
class Record:
    """One input row, normalized so downstream stages never branch on format.

    For TSV the audio path is derived from ``segment_id``; for NeMo manifests
    the path is authoritative and ``segment_id`` is derived from its stem.
    Both directions are resolved here so later stages see one shape.
    """

    segment_id: str
    text: str
    audio_filepath: Optional[Path] = None
    duration: Optional[float] = None


def normalize_text(text: str) -> str:
    """Normalize to Unicode NFKC.

    NFKC, not NFC, is what Thai needs here. สระอำ encodes either as U+0E33 or
    as นิคหิต + สระอา (U+0E4D U+0E32); the two render identically but compare
    unequal, so without folding them a correct prediction is scored as an
    error. U+0E33's decomposition is <compat>, which NFC and NFD both ignore —
    only NFKC maps ำ -> ํา.

    NFKC is broader than NFC (it also folds full-width forms and ligatures),
    which is acceptable for text destined for WER comparison.
    """
    return unicodedata.normalize(NORMAL_FORM, text)


def segment_id_from_path(audio_filepath: Path) -> str:
    """Derive a segment id from an audio file path (its stem)."""
    return audio_filepath.stem


def shard_of(segment_id: str) -> str:
    """Return the checkpoint shard key: the first segment-id component.

    This is also the first audio directory level, so checkpoint locality
    matches on-disk locality.
    """
    head: str = segment_id.split("-", 1)[0]
    if not head:
        raise MalformedSegmentIdError(f"empty shard component in {segment_id!r}")
    return head


def audio_path_for_segment(segment_id: str, audio_root: Path) -> Path:
    """Map ``148-148801-11`` to ``<audio_root>/train/148/148801/148-148801-11.wav``."""
    parts: List[str] = segment_id.split("-")
    if len(parts) < SEGMENT_ID_PARTS or not all(parts[:2]):
        raise MalformedSegmentIdError(
            f"segment id {segment_id!r} does not split into "
            f"{SEGMENT_ID_PARTS} '-'-separated parts"
        )
    return audio_root / AUDIO_SUBDIR / parts[0] / parts[1] / f"{segment_id}{AUDIO_SUFFIX}"


def is_valid_segment_id(segment_id: str) -> bool:
    """Cheap shape check used when replaying checkpoint lines."""
    if not segment_id or segment_id.isspace():
        return False
    parts: List[str] = segment_id.split("-")
    return len(parts) >= SEGMENT_ID_PARTS and all(parts[:2])
