"""Whisper model wrapper.

``import transformers`` happens inside :meth:`AsrModelWrapper.load`, never at
module import. That is what lets the test suite and ``--dry-run`` work on a
machine without the heavy dependency installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import (Any, List, Optional, Sequence)

from .audio import load_samples
from .config import (AudioConfig, ModelConfig)

LOGGER: logging.Logger = logging.getLogger(__name__)

TRANSFORMERS_HELP: str = (
    "transformers is required for the Whisper backend. Run "
    "`uv sync --frozen`, or ./setup_and_run.sh --setup-only for a full "
    "environment build."
)


class ModelLoadError(RuntimeError):
    """Raised when the model or its dependencies cannot be prepared."""


def _resolve_dtype(name: str) -> Any:
    """Map a config dtype string to a torch dtype."""
    import torch

    mapping: dict[str, Any] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ModelLoadError(
            f"unsupported model.dtype {name!r}; expected one of {sorted(mapping)}"
        )
    return mapping[name]


class AsrModelWrapper:
    """Thin wrapper over the Whisper model implementing the transcribe call."""

    def __init__(self, cfg: ModelConfig, device: str,
                 audio_cfg: Optional[AudioConfig] = None) -> None:
        self.cfg: ModelConfig = cfg
        self.device: str = device
        self.audio_cfg: Optional[AudioConfig] = audio_cfg
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            from transformers import (WhisperForConditionalGeneration,
                                      WhisperProcessor)
        except ImportError as exc:
            raise ModelLoadError(f"could not import transformers. {TRANSFORMERS_HELP}") from exc
        except OSError as exc:
            # A native library failed to load (e.g. a partially extracted CUDA
            # wheel). Distinct from a missing package, and worth saying so.
            raise ModelLoadError(
                f"transformers failed to import a shared library: {exc}. "
                "The nvidia-* wheels may be incomplete or off LD_LIBRARY_PATH; "
                "./setup_and_run.sh --setup-only repairs both."
            ) from exc

        source: str = (
            str(self.cfg.local_model_path)
            if self.cfg.local_model_path is not None
            else self.cfg.hf_repo_id
        )
        LOGGER.info("loading %s on %s (dtype=%s)", source, self.device, self.cfg.dtype)

        try:
            dtype: Any = _resolve_dtype(self.cfg.dtype)
            processor: Any = WhisperProcessor.from_pretrained(source)
            model: Any = WhisperForConditionalGeneration.from_pretrained(
                source, dtype=dtype
            )
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any load failure with context
            raise ModelLoadError(
                f"could not load {source}: {exc}. If the repo is gated, set HF_TOKEN. "
                "A truncated download also presents this way — check disk space."
            ) from exc

        model = model.to(self.device)
        model.eval()

        self._processor = processor
        self._model = model
        self._dtype = dtype
        LOGGER.info(
            "model ready (language=%s task=%s)", self.cfg.target_lang, self.cfg.task
        )

    def transcribe_batch(self, audio_paths: Sequence[Path]) -> List[str]:
        """Transcribe a batch, returning one string per input path."""
        if self._model is None or self._processor is None:
            raise ModelLoadError("transcribe_batch called before load()")
        if not audio_paths:
            return []

        import torch

        # Whisper's processor consumes arrays, not paths.
        waveforms: List[Any] = [
            load_samples(path, self.audio_cfg) for path in audio_paths
        ]

        features: Any = self._processor(
            waveforms,
            sampling_rate=self.cfg.sampling_rate,
            return_tensors="pt",
        ).input_features
        features = features.to(self.device, self._dtype)

        with torch.no_grad():
            generated: Any = self._model.generate(
                features,
                language=self.cfg.target_lang,
                task=self.cfg.task,
                max_new_tokens=self.cfg.max_new_tokens,
            )

        texts: List[str] = [
            str(text).strip()
            for text in self._processor.batch_decode(generated, skip_special_tokens=True)
        ]

        if len(texts) != len(audio_paths):
            raise RuntimeError(
                f"model returned {len(texts)} result(s) for {len(audio_paths)} input(s)"
            )
        return texts
