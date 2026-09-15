"""Segment-id decomposition, path derivation, and NFC normalization."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from thai_asr_batch.records import (MalformedSegmentIdError, Record,
                                    audio_path_for_segment, is_valid_segment_id,
                                    normalize_text, segment_id_from_path, shard_of)

from conftest import (THAI_COMPOSED, THAI_DECOMPOSED)


def test_audio_path_follows_the_nested_rule() -> None:
    root: Path = Path("/data/th")
    assert audio_path_for_segment("148-148801-11", root) == (
        root / "train" / "148" / "148801" / "148-148801-11.wav"
    )


def test_audio_path_uses_full_id_as_filename() -> None:
    result: Path = audio_path_for_segment("100-100000-0", Path("/x"))
    assert result.name == "100-100000-0.wav"
    assert result.parent.name == "100000"
    assert result.parent.parent.name == "100"


@pytest.mark.parametrize("bad_id", ["", "148", "148-148801", "-148801-11", "148--11"])
def test_malformed_ids_raise(bad_id: str) -> None:
    with pytest.raises(MalformedSegmentIdError):
        audio_path_for_segment(bad_id, Path("/x"))


def test_shard_is_the_first_component() -> None:
    assert shard_of("148-148801-11") == "148"


def test_segment_id_from_path_is_the_stem() -> None:
    assert segment_id_from_path(Path("/a/b/148-148801-11.wav")) == "148-148801-11"


def test_is_valid_segment_id() -> None:
    assert is_valid_segment_id("148-148801-11")
    assert not is_valid_segment_id("148-148801")
    assert not is_valid_segment_id("")
    assert not is_valid_segment_id("   ")


def test_normalization_collapses_the_sara_am_encodings() -> None:
    """สระอำ in either spelling must compare equal after normalization.

    This is the case that motivated the requirement: without folding these,
    a correct prediction gets scored as an error.
    """
    assert THAI_COMPOSED != THAI_DECOMPOSED
    assert normalize_text(THAI_COMPOSED) == normalize_text(THAI_DECOMPOSED)


def test_nfc_alone_would_not_have_worked() -> None:
    """Pin why this is NFKC: U+0E33 decomposes only under <compat>.

    NFC and NFD both leave BOTH spellings untouched, so they never converge.
    Guards against someone "simplifying" normalize_text back to NFC.
    """
    assert unicodedata.decomposition("ำ").startswith("<compat>")
    assert (
        unicodedata.normalize("NFC", THAI_COMPOSED)
        != unicodedata.normalize("NFC", THAI_DECOMPOSED)
    )
    assert (
        unicodedata.normalize("NFD", THAI_COMPOSED)
        != unicodedata.normalize("NFD", THAI_DECOMPOSED)
    )


def test_normalization_handles_canonical_compositions() -> None:
    """Ordinary canonical differences fold too (NFKC subsumes NFC)."""
    assert normalize_text("é") == "é"


def test_normalization_is_idempotent() -> None:
    assert normalize_text(normalize_text(THAI_DECOMPOSED)) == normalize_text(THAI_DECOMPOSED)


def test_normalization_leaves_conforming_text_untouched() -> None:
    text: str = "ให้ทันภายใน24พฤศจิกายนนี้"
    assert normalize_text(text) == text


def test_record_defaults() -> None:
    record: Record = Record(segment_id="100-100000-0", text="x")
    assert record.audio_filepath is None
    assert record.duration is None
