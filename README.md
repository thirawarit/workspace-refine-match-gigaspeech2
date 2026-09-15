# Thai ASR Batch Inference — GigaSpeech2 refine-match

Runs [`typhoon-ai/typhoon-asr-streaming-nemotron-0.6b`](https://huggingface.co/typhoon-ai/typhoon-asr-streaming-nemotron-0.6b)
over GigaSpeech2 Thai transcripts and produces a side-by-side file of reference vs. predicted
text for WER scoring and refinement diffing.

See [`SPEC.md`](SPEC.md) for the full design and rationale.

## Quick start

```bash
./setup_and_run.sh --setup-only                      # build the environment once

./setup_and_run.sh validate   --input data/train_refined.tsv --config configs/vps_h100.yaml
./setup_and_run.sh transcribe --input data/train_refined.tsv --config configs/vps_h100.yaml
./setup_and_run.sh combine    --input data/train_refined.tsv --config configs/vps_h100.yaml
```

`run` does transcribe-then-combine in one go.

## Data flow

```
train_refined.tsv                 segment_id <TAB> text          (headerless)
        │
        ├─ transcribe ──▶ predicted-train_refined.tsv            (same format)
        │
        └─ combine ─────▶ combined-train_refined.tsv             (with header)
                              segment_id, orig_text, pred_text
```

Audio paths are derived from the segment id:

```
148-148801-11  →  <audio_root>/train/148/148801/148-148801-11.wav
```

JSONL input is NeMo-manifest shaped (`audio_filepath`/`text`/`duration`); the `segment_id` is
derived from the path stem so both formats join identically.

## Text normalization

Both `orig_text` and `pred_text` are normalized to **Unicode NFKC** in the combined file.

Thai สระอำ has two encodings — U+0E33, or นิคหิต + สระอา (U+0E4D U+0E32) — that render
identically but compare unequal. NFC does *not* fold them: U+0E33's decomposition is tagged
`<compat>`, so NFC and NFD both leave either spelling alone. Only NFKC maps ำ → ํา. Without it,
WER would count correct predictions as errors.

## The NeMo dependency

**NeMo is deliberately not in `requirements.txt`.** Stock `nemo_toolkit` does not contain
`EncDecRNNTBPEModelWithPrompt`, which this model requires. It must be built from source at
pinned commit `907edfd`:

```bash
git clone https://github.com/NVIDIA/NeMo && cd NeMo
git checkout 907edfd && pip install -e .
export NEMO_ROOT=/path/to/NeMo
```

`setup_and_run.sh` does this automatically and verifies the commit on every run. A drifted
checkout is the most likely cause of a mysterious missing-class error, so the script aborts
rather than continuing.

## Commands

| Command | Purpose |
|---|---|
| `transcribe` | Run ASR → `predicted-<name>.<ext>` |
| `combine` | Join input + predictions on `segment_id` |
| `run` | `transcribe` then `combine` |
| `validate` | Check config and audio paths without loading the model |
| `status` | Report resume progress without attaching to the running process |

Useful flags: `--limit N`, `--dry-run`, `--no-resume`, `--force-resume`, `--batch-size`,
`--device`, `--low-memory`. CLI flags override YAML.

## Configuration

| Profile | Use |
|---|---|
| `configs/default.yaml` | Baseline; all keys documented inline |
| `configs/vps_h100.yaml` | Production run — CUDA index 0, 12 IO workers |
| `configs/local_cpu.yaml` | Laptop smoke tests — CPU, batch size 2 |

MPS stays off by default: NeMo RNN-T on Apple MPS tends to fail outright rather than degrade,
so it requires an explicit `allow_mps: true`.

## Long runs

The full Thai `train_refined` split is ~10,262 hours — roughly **10 days** on one H100. Run it
detached and check in with `status`:

```bash
tmux new -s asr
./setup_and_run.sh transcribe --input data/train_refined.tsv --config configs/vps_h100.yaml
# detach with Ctrl-B D

./setup_and_run.sh --skip-setup status --input data/train_refined.tsv --config configs/vps_h100.yaml
```

Restarting is safe and cheap: progress is checkpointed per shard, so a killed run resumes where
it stopped. Use `--skip-setup` on restarts to avoid re-resolving dependencies.

**`status` reports the empty-prediction rate.** If most predictions are empty, stop and check
the model and GPU — that should be caught in hour 1, not on day 10.

## Failure handling

A clip that cannot be read or transcribed gets an empty `pred_text`, a line in
`logs/errors-<stamp>.jsonl`, and the run continues. The combined file stays 1:1 with the input
so downstream scoring can align rows.

If an entire batch fails repeatedly (usually an unusable CUDA context after an OOM), the run
aborts instead of writing empty rows for hours.

## Development

Dependencies are managed with [uv](https://docs.astral.sh/uv/) in **project mode**:
`pyproject.toml` declares them, `uv.lock` pins the exact resolved versions, and both are
committed. `.python-version` pins the interpreter to 3.12.

```bash
uv sync                 # create/refresh .venv from the lockfile
uv run pytest tests/ -q # run the suite
```

Adding a dependency:

```bash
uv add <package>          # runtime
uv add --dev <package>    # tests/tooling only
uv lock --check           # verify the lock matches pyproject.toml before committing
```

Never use `uv pip install` or hand-edit `pyproject.toml` dependency entries — both bypass the
lockfile and break reproducibility. The one exception is NeMo, which cannot be expressed in
`uv.lock` at all (see above) and is installed from source by `setup_and_run.sh`.

The suite needs **no GPU and no NeMo** — `import nemo` is deferred into `model.load()`, which
the tests never reach. Fixtures generate real tiny WAVs via the stdlib `wave` module.

## Layout

```
src/thai_asr_batch/
├── config.py         # typed config + device resolution
├── logging_utils.py  # Bangkok-timezone logging
├── records.py        # the normal form + id/path rules
├── io_formats.py     # TSV/JSONL readers and writers
├── audio.py          # probing + resample-on-the-fly
├── model.py          # NeMo wrapper (deferred import)
├── batching.py       # length bucketing + failure isolation
├── checkpoint.py     # sharded resume state
├── inference.py      # orchestrator
├── combine.py        # the join
└── cli.py            # argparse entry point
```
