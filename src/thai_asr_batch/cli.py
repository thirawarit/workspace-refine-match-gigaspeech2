"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import (Any, Dict, List, Optional)

from .checkpoint import (CheckpointMismatchError, CheckpointStore)
from .combine import (CombineStats, combine, combine_low_memory)
from .config import (AppConfig, load_config, resolve_device)
from .io_formats import (detect_format, predicted_output_path)
from .inference import (InferenceStats, run_inference)
from .logging_utils import (get_logger, setup_logging)

LOGGER: logging.Logger = get_logger(__name__)

DEFAULT_CONFIG: Path = Path("configs/default.yaml")
COMBINED_PREFIX: str = "combined-"


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="thai-asr-batch",
        description="Batch ASR inference over GigaSpeech2 Thai transcripts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                            help="YAML config file (default: configs/default.yaml)")
        target.add_argument("--audio-root", type=Path, default=None,
                            help="override paths.audio_root")
        target.add_argument("--output-dir", type=Path, default=None,
                            help="override paths.output_dir")
        target.add_argument("--log-level", type=str, default=None,
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                            help="override runtime.log_level")

    transcribe = subparsers.add_parser("transcribe", help="run ASR over an input file")
    add_common(transcribe)
    transcribe.add_argument("--input", type=Path, required=True, help="TSV or JSONL input")
    transcribe.add_argument("--limit", type=int, default=None,
                            help="stop after N un-transcribed rows")
    transcribe.add_argument("--dry-run", action="store_true",
                            help="validate audio paths without loading the model")
    transcribe.add_argument("--no-resume", action="store_true",
                            help="ignore checkpoints and start over")
    transcribe.add_argument("--force-resume", action="store_true",
                            help="resume even if the input no longer matches the checkpoint")
    transcribe.add_argument("--batch-size", type=int, default=None,
                            help="override batch.batch_size")
    transcribe.add_argument("--device", type=str, default=None,
                            help="override device preference, e.g. cuda/cpu/mps")

    combine_parser = subparsers.add_parser("combine", help="join input with predictions")
    add_common(combine_parser)
    combine_parser.add_argument("--input", type=Path, required=True)
    combine_parser.add_argument("--predicted", type=Path, default=None,
                                help="defaults to predicted-<input name> in output_dir")
    combine_parser.add_argument("--output", type=Path, default=None,
                                help="defaults to combined-<input name> in output_dir")
    combine_parser.add_argument("--format", dest="out_format", type=str, default=None,
                                choices=["tsv", "jsonl"])
    combine_parser.add_argument("--low-memory", action="store_true",
                                help="sort-merge join instead of an in-RAM hash join")

    run_parser = subparsers.add_parser("run", help="transcribe then combine")
    add_common(run_parser)
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--no-resume", action="store_true")
    run_parser.add_argument("--force-resume", action="store_true")
    run_parser.add_argument("--batch-size", type=int, default=None)
    run_parser.add_argument("--device", type=str, default=None)
    run_parser.add_argument("--low-memory", action="store_true")

    validate = subparsers.add_parser("validate", help="check config and audio paths")
    add_common(validate)
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--limit", type=int, default=None)

    status = subparsers.add_parser("status", help="report resume progress")
    add_common(status)
    status.add_argument("--input", type=Path, required=True)

    return parser


def _overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    paths: Dict[str, Any] = {}
    if getattr(args, "audio_root", None) is not None:
        paths["audio_root"] = str(args.audio_root)
    if getattr(args, "output_dir", None) is not None:
        paths["output_dir"] = str(args.output_dir)
    if paths:
        overrides["paths"] = paths

    if getattr(args, "batch_size", None) is not None:
        overrides["batch"] = {"batch_size": args.batch_size}
    if getattr(args, "device", None) is not None:
        overrides["device"] = {"prefer": [args.device]}
    if getattr(args, "log_level", None) is not None:
        overrides["runtime"] = {"log_level": args.log_level}
    return overrides


def main(argv: Optional[List[str]] = None) -> int:
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args(argv)

    if not args.config.exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2

    cfg: AppConfig = load_config(args.config, _overrides_from_args(args))
    level: int = getattr(logging, cfg.runtime.log_level.upper(), logging.INFO)
    log_path: Path = setup_logging(cfg.paths.log_dir, level=level)
    LOGGER.info("session log: %s", log_path)

    if not args.input.exists():
        LOGGER.error("input not found: %s", args.input)
        return 2

    try:
        if args.command == "transcribe":
            return _cmd_transcribe(args, cfg)
        if args.command == "combine":
            return _cmd_combine(args, cfg)
        if args.command == "run":
            code: int = _cmd_transcribe(args, cfg)
            return code if code != 0 else _cmd_combine(args, cfg)
        if args.command == "validate":
            return _cmd_validate(args, cfg)
        if args.command == "status":
            return _cmd_status(args, cfg)
    except CheckpointMismatchError as exc:
        LOGGER.error("%s", exc)
        return 3
    except KeyboardInterrupt:
        LOGGER.warning("interrupted; checkpoints are durable, re-run to resume")
        return 130

    parser.print_help()
    return 2


def _cmd_transcribe(args: argparse.Namespace, cfg: AppConfig) -> int:
    stats: InferenceStats = run_inference(
        input_path=args.input,
        cfg=cfg,
        limit=getattr(args, "limit", None),
        dry_run=getattr(args, "dry_run", False),
        resume=not getattr(args, "no_resume", False),
        force_resume=getattr(args, "force_resume", False),
    )
    return 0 if stats.failed == 0 or stats.succeeded > 0 else 1


def _cmd_combine(args: argparse.Namespace, cfg: AppConfig) -> int:
    predicted: Path = (
        getattr(args, "predicted", None)
        or predicted_output_path(args.input, cfg.paths.output_dir)
    )
    if not predicted.exists():
        LOGGER.error("predictions not found: %s", predicted)
        return 2

    output: Path = (
        getattr(args, "output", None)
        or cfg.paths.output_dir / f"{COMBINED_PREFIX}{args.input.name}"
    )
    out_format = getattr(args, "out_format", None)

    if getattr(args, "low_memory", False):
        stats: CombineStats = combine_low_memory(
            args.input, predicted, output, cfg, output_format=out_format
        )
    else:
        stats = combine(args.input, predicted, output, cfg, output_format=out_format)

    LOGGER.info("wrote %s", output)
    return 0 if stats.total_input > 0 else 1


def _cmd_validate(args: argparse.Namespace, cfg: AppConfig) -> int:
    LOGGER.info("config=%s", args.config)
    LOGGER.info("audio_root=%s (exists=%s)",
                cfg.paths.audio_root, cfg.paths.audio_root.exists())
    LOGGER.info("device would be: %s", resolve_device(cfg.device))
    LOGGER.info("input format: %s", detect_format(args.input))

    stats: InferenceStats = run_inference(
        input_path=args.input,
        cfg=cfg,
        limit=getattr(args, "limit", None),
        dry_run=True,
        resume=False,
    )
    if stats.skipped_missing_audio:
        LOGGER.error(
            "%d row(s) have missing or unreadable audio — fix before a real run",
            stats.skipped_missing_audio,
        )
        return 1
    LOGGER.info("validation passed")
    return 0


def _cmd_status(args: argparse.Namespace, cfg: AppConfig) -> int:
    store: CheckpointStore = CheckpointStore(
        root=cfg.paths.checkpoint_dir,
        input_path=args.input,
        cfg=cfg.checkpoint,
        config_fingerprint=cfg.fingerprint(),
        input_format=detect_format(args.input),
    )
    if not store.dir.exists():
        LOGGER.info("no checkpoint state at %s; nothing started yet", store.dir)
        return 0

    completed: int = store.count_completed()
    LOGGER.info("checkpoint dir: %s", store.dir)
    LOGGER.info("completed segments: %d", completed)

    predicted: Path = predicted_output_path(args.input, cfg.paths.output_dir)
    if predicted.exists():
        size_mb: float = predicted.stat().st_size / (1024.0 * 1024.0)
        empty: int = _count_empty_predictions(predicted)
        LOGGER.info("predictions: %s (%.1f MB)", predicted, size_mb)
        # An all-empty run should be caught in hour 1, not on day 10.
        if completed and empty / max(1, completed) > 0.5:
            LOGGER.error(
                "%d of %d predictions are empty — check the model/GPU immediately",
                empty, completed,
            )
        else:
            LOGGER.info("empty predictions: %d", empty)
    return 0


def _count_empty_predictions(path: Path) -> int:
    empty: int = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped: str = line.rstrip("\n")
            if not stripped:
                continue
            _, _, text = stripped.partition("\t")
            if not text.strip():
                empty += 1
    return empty


if __name__ == "__main__":
    raise SystemExit(main())
