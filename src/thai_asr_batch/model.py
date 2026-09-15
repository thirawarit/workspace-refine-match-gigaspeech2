"""NeMo model wrapper.

``import nemo`` happens inside :meth:`AsrModelWrapper.load`, never at module
import. That is what lets the test suite and ``--dry-run`` work on a machine
with NeMo uninstalled — which is the whole local development story here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import (Any, List, Optional, Sequence)

from .config import ModelConfig

LOGGER: logging.Logger = logging.getLogger(__name__)

PINNED_NEMO_COMMIT: str = "907edfd"
NEMO_HELP: str = (
    "Stock nemo_toolkit lacks EncDecRNNTBPEModelWithPrompt. The supported fix is "
    "./setup_and_run.sh --setup-only, which clones NeMo at the pinned commit, "
    "installs the required extras, routes wheel unpacking off /tmp and puts the "
    "CUDA libraries on the loader path.\n"
    f"By hand: git clone https://github.com/NVIDIA/NeMo && cd NeMo && git checkout "
    f"{PINNED_NEMO_COMMIT} && uv pip install -e '.[asr,cu13]'\n"
    "The [asr] extra is REQUIRED: a bare `-e .` omits hydra-core, omegaconf and "
    "lightning, and the import then fails with \"No module named 'hydra'\". "
    "`asr-only` is not a substitute — it excludes hydra. Use cu12 in place of "
    "cu13 on a CUDA 12 host."
)


class ModelLoadError(RuntimeError):
    """Raised when the model or its NeMo dependency cannot be prepared."""


def resolve_model_file(cfg: ModelConfig) -> Path:
    """Return a local ``.nemo`` path, downloading from HuggingFace if needed."""
    if cfg.local_model_path is not None:
        if not cfg.local_model_path.exists():
            raise ModelLoadError(f"local_model_path does not exist: {cfg.local_model_path}")
        LOGGER.info("using local checkpoint %s", cfg.local_model_path)
        return cfg.local_model_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ModelLoadError(
            "huggingface_hub is required to download the checkpoint; "
            "run `uv sync --frozen` (requirements.txt is only a pointer stub — "
            "pyproject.toml and uv.lock are the source of truth)"
        ) from exc

    LOGGER.info("resolving %s from %s", cfg.nemo_filename, cfg.hf_repo_id)
    try:
        downloaded: str = hf_hub_download(
            repo_id=cfg.hf_repo_id,
            filename=cfg.nemo_filename,
        )
    except Exception as exc:  # noqa: BLE001 - surface any hub failure with context
        raise ModelLoadError(
            f"could not download {cfg.nemo_filename} from {cfg.hf_repo_id}: {exc}. "
            "If the repo is gated, set HF_TOKEN."
        ) from exc

    LOGGER.info("checkpoint at %s", downloaded)
    return Path(downloaded)


class AsrModelWrapper:
    """Thin wrapper over the NeMo ASR model implementing the transcribe call."""

    def __init__(self, cfg: ModelConfig, device: str) -> None:
        self.cfg: ModelConfig = cfg
        self.device: str = device
        self._model: Optional[Any] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise ModelLoadError(f"could not import nemo.collections.asr. {NEMO_HELP}") from exc

        model_path: Path = resolve_model_file(self.cfg)
        LOGGER.info("restoring model on %s", self.device)
        try:
            model: Any = nemo_asr.models.ASRModel.restore_from(
                str(model_path), map_location=self.device
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"restore_from failed for {model_path}: {exc}. {NEMO_HELP}") from exc

        model.eval()
        self._apply_prompt(model)
        self._apply_strip_lang_tags(model)

        self._model = model
        LOGGER.info("model ready (target_lang=%s)", self.cfg.target_lang)

    def _apply_prompt(self, model: Any) -> None:
        # hasattr-guarded so NeMo version drift fails loudly with the commit
        # hash, instead of an opaque AttributeError deep into a run.
        if not hasattr(model, "set_inference_prompt"):
            raise ModelLoadError(f"model has no set_inference_prompt(). {NEMO_HELP}")
        model.set_inference_prompt(self.cfg.target_lang)

    def _apply_strip_lang_tags(self, model: Any) -> None:
        if not self.cfg.strip_lang_tags:
            return
        decoding: Any = getattr(model, "decoding", None)
        if decoding is None or not hasattr(decoding, "set_strip_lang_tags"):
            raise ModelLoadError(f"model.decoding has no set_strip_lang_tags(). {NEMO_HELP}")
        decoding.set_strip_lang_tags(True)

    def transcribe_batch(self, audio_paths: Sequence[Path]) -> List[str]:
        """Transcribe a batch, returning one string per input path."""
        if self._model is None:
            raise ModelLoadError("transcribe_batch called before load()")
        if not audio_paths:
            return []

        raw: Any = self._model.transcribe(
            audio=[str(item) for item in audio_paths],
            target_lang=self.cfg.target_lang,
            return_hypotheses=True,
        )
        texts: List[str] = _normalize_hypotheses(raw)

        if len(texts) != len(audio_paths):
            raise RuntimeError(
                f"model returned {len(texts)} result(s) for {len(audio_paths)} input(s)"
            )
        return texts


def _normalize_hypotheses(raw: Any) -> List[str]:
    """Flatten NeMo's return shape.

    Versions have returned both ``List[Hypothesis]`` and nested
    ``List[List[Hypothesis]]`` (best-first), so unwrap one level when present.
    """
    if raw is None:
        return []

    items: List[Any] = list(raw)
    if items and isinstance(items[0], (list, tuple)):
        items = [group[0] if len(group) else None for group in items]

    texts: List[str] = []
    for item in items:
        if item is None:
            texts.append("")
        elif hasattr(item, "text"):
            texts.append(str(item.text))
        else:
            texts.append(str(item))
    return texts
