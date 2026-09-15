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

Audio paths are derived from the segment id. **`audio_root` is the parent of `train/`** — the
pipeline appends the `train` segment itself, so pointing it at the `train/` directory builds
`.../train/train/...` and finds nothing:

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

## Secrets

```bash
cp .env.example .env
chmod 600 .env
# then edit .env and set HF_TOKEN=hf_...
```

`setup_and_run.sh` sources `.env` at startup and exports its contents, so the token reaches both
`uv sync` and the run. `.env` is gitignored; `.env.example` is the committed template.

`HF_TOKEN` is only required if the model repo is gated or private — a public repo downloads
without it. Create one at <https://huggingface.co/settings/tokens>; read scope is sufficient.

Running Python directly rather than through the script? Load it yourself first:

```bash
set -a && source .env && set +a
```

## Install-time temp space

Large CUDA wheels (torch, cuDNN, cuBLAS, NCCL) unpack to a temp directory before landing in the
venv. The system default `/tmp` is often a small partition — sometimes a RAM-backed tmpfs — and
running out of space there produces a **silently partial install**: some `.so` files present,
the rest missing. It surfaces much later as a baffling `ImportError`, e.g.
`libcudnn.so.9: cannot open shared object file`, long after the install reported success.

`setup_and_run.sh` therefore points both temp locations at the repo's own disk and refuses to
install without enough headroom:

| Variable | Default | Purpose |
|---|---|---|
| `SCRATCH_ROOT` | `<repo>` | Base for the two below |
| `ASR_TMPDIR` | `<repo>/.tmp` | Where wheels unpack — becomes `TMPDIR` |
| `ASR_UV_CACHE_DIR` | `<repo>/.uv-cache` | uv's download cache (uv prefers this over `TMPDIR`) |
| `MIN_FREE_MB` | `15000` | Abort below this much free space |

The script **overrides any inherited `TMPDIR`** rather than deferring to it. macOS presets
`TMPDIR`, and most Linux shells inherit `/tmp`, so a `${TMPDIR:-default}` fallback would always
keep the inherited value and never take effect — leaving you on exactly the small partition this
is meant to avoid. Use `ASR_TMPDIR` to choose a different location deliberately.

If the repo itself lives on a small volume, redirect them:

```bash
SCRATCH_ROOT=/data/big ./setup_and_run.sh --setup-only
```

Recovering from an already-partial install:

```bash
export TMPDIR=/data/big/tmp UV_CACHE_DIR=/data/big/uv-cache
uv pip install --force-reinstall --no-cache nvidia-cudnn-cu13
```

`--no-cache` matters: without it a corrupt cached wheel is simply re-extracted.

## The NeMo dependency

The ASR backend is [`typhoon-ai/typhoon-whisper-medium`](https://huggingface.co/typhoon-ai/typhoon-whisper-medium),
a Whisper-medium fine-tune (0.8B params) loaded through `transformers`. It installs straight
from the lockfile — no source build, no pinned commit, no optional extras.

```python
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16)
ids = model.generate(feats, language="th", task="transcribe", max_new_tokens=440)
```

Two constraints worth knowing:

- **`language="th"`**, ISO-639-1 — not the BCP-47 `"th-TH"`.
- **A hard 30-second encoder window.** Longer clips are silently truncated to the first window
  by the feature extractor, returning partial text that reads like a poor transcription rather
  than an error. The pipeline detects these via `audio.max_duration_seconds`, logs them to the
  error journal, and still transcribes the first window so no row is lost. `validate` reports
  the count up front, so the real exposure is known before GPU time is spent.

The checkpoint is several GB and downloads to `HF_HOME`, which `setup_and_run.sh` points at
`<repo>/.hf-cache` rather than `~/.cache` — see the temp-space section above.

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
`--device`, `--cuda-index`, `--low-memory`. CLI flags override YAML.

### Choosing a GPU

`--device` takes `cpu`, `mps`, `cuda` or `cuda:N`; `--cuda-index N` sets the GPU index on its
own, leaving the YAML preference list intact. Both are available on every subcommand, so
`validate` previews the choice without loading the model:

```bash
./setup_and_run.sh validate --input train.tsv --device cuda:1   # which GPU would be used?
./setup_and_run.sh transcribe --input train.tsv --cuda-index 1
```

A device named on the command line is **never** silently downgraded: if that GPU is missing or
the index is out of range, the run aborts instead of spending days on CPU. A YAML `prefer` list
keeps the soft fallback, so `["cuda", "cpu"]` still degrades gracefully on a CPU-only box.

Note that `CUDA_VISIBLE_DEVICES` renumbers devices — under a mask of `1` the only visible GPU is
`cuda:0`. Set the mask or the index, not both.

## Configuration

| Profile | Use |
|---|---|
| `configs/default.yaml` | Baseline; all keys documented inline |
| `configs/vps_h100.yaml` | Production run — CUDA index 0, 12 IO workers |
| `configs/local_cpu.yaml` | Laptop smoke tests — CPU, batch size 2 |

MPS stays off by default: Whisper generation on Apple MPS is slow and historically flaky, so it
requires an explicit `allow_mps: true`.

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
