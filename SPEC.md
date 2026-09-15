# SPEC — Thai ASR Batch-Inference Pipeline (GigaSpeech2 refine-match)

## 1. Purpose

Run the Typhoon Thai ASR model over GigaSpeech2 Thai transcripts and produce a side-by-side
file pairing each segment's reference transcript with the model's prediction, for downstream
WER scoring and refinement diffing.

Target corpus is the full GigaSpeech2 Thai `train_refined` split: ~10,262 hours, approximately
**10 days of single-GPU compute** at the model card's ~42x realtime. Multi-day runtime makes
crash-resume and per-clip failure isolation core requirements rather than polish.

## 2. Functional requirements

| # | Requirement |
|---|---|
| R1 | Use model `typhoon-ai/typhoon-whisper-medium` (Whisper-medium fine-tune, encoder-decoder). |
| R2 | Accept TSV or JSONL input; emit predictions in the same format. `AbC.tsv` → `predicted-AbC.tsv`. |
| R3 | Combine input + predictions into a new file with columns `segment_id`, `orig_text`, `pred_text`. |

## 3. Data contract

### 3.1 TSV input (GigaSpeech2 native)

Headerless, two tab-separated columns: `segment_id`, `text`.

```tsv
100-100000-0	ท่านผู้ชมครับเรื่องของ COVID-19 วันนี้ไทยพบผู้ป่วยเพิ่มหนึ่งคนนะครับ
148-148801-11	ให้ทันภายใน24พฤศจิกายนนี้
```

### 3.2 JSONL input (NeMo manifest)

`{"audio_filepath": ..., "text": ..., "duration": ...}`. Manifests carry no `segment_id`, so it
is derived from the `audio_filepath` stem. This keeps the combine step keyed on `segment_id`
for **both** formats.

### 3.3 Audio path rule

`segment_id` splits on `-` into three parts; parts[0] and parts[1] are directory levels and the
full id is the filename:

```
148-148801-11  →  <audio_root>/train/148/148801/148-148801-11.wav
```

Malformed ids (fewer than 3 parts) raise `MalformedSegmentIdError`, which the loader converts
into a skip-with-log rather than a crash.

### 3.4 Audio format

Natively 16 kHz mono 16-bit PCM WAV. Resample-on-the-fly is enabled defensively but implemented
as a no-op fast path for conforming files (§6.3).

### 3.5 Output files

- **Interim**: `predicted-<name>.<ext>`, same format as input, full passthrough.
- **Combined**: 3-column TSV **with a header** — `segment_id`, `orig_text`, `pred_text`. Unlike
  the input this file is for human/eval consumption, and named columns imply a header.

## 4. Text normalization

Both `pred_text` and `orig_text` are normalized to **Unicode NFKC**.

This is non-negotiable for Thai. สระอำ encodes either as U+0E33 or as นิคหิต + สระอา
(U+0E4D U+0E32); the two render identically but compare unequal byte-wise. Without folding them,
correct predictions would be scored as errors.

**NFKC, not NFC.** U+0E33's decomposition is tagged `<compat>`, which the canonical forms ignore
— NFC and NFD both leave *both* spellings untouched, so they never converge. Only NFKC maps
ำ → ํา. This was verified empirically, not assumed, and `tests/test_records.py` pins it so the
choice cannot be silently "simplified" back to NFC.

NFKC is broader than NFC (it also folds full-width forms and ligatures), which is acceptable for
text whose purpose is WER comparison. Beyond normalization the model output is left raw —
further cleanup is lossy and belongs in a separate, tunable scoring step.

## 5. Environment

### 5.1 Hardware targets

| | VPS (primary) | Local (dev) |
|---|---|---|
| OS | Linux, CUDA 13.0 | macOS, Apple M4 Pro |
| Compute | 2× H100 80GB — **index 0 only** (index 1 has 0 MiB free) | MPS/CPU |
| RAM | 503 GB | 24 GB |
| Python | 3.12 | **3.9.6 on PATH** (not 3.10.20) |

### 5.2 Device policy

`cuda → cpu`. MPS is selected **only** behind an explicit `allow_mps` opt-in: Whisper on Apple
MPS is slow and historically flaky for long generations, and a silent fallback would turn a
smoke test into a confusing hang. The runner also
exports `CUDA_VISIBLE_DEVICES=0` so the index-0 constraint holds belt-and-braces.

**Device specs.** `device.prefer` entries are `cpu`, `mps`, `cuda` or `cuda:N`. A bare `cuda`
uses `device.cuda_index`; `cuda:N` overrides it. `--device` and `--cuda-index` expose both from
the CLI on every subcommand, so `validate` previews the choice without loading the model.

**Explicit requests fail loudly.** A device named on the command line never degrades to CPU: an
unavailable GPU, an out-of-range index, or an unparseable spec raises `DeviceConfigError` (exit
code 4). A YAML `prefer` list keeps the soft fallback. This closes a real defect — `resolve_device`
compared against the bare strings `"cuda"`/`"mps"`/`"cpu"`, so *any* indexed spec matched nothing,
fell through the loop, and hit the CPU fallback. `--device cuda:1` silently ran on CPU, which on a
ten-day corpus is the single most expensive way this pipeline can fail.

Index validation also detects the `CUDA_VISIBLE_DEVICES` trap: masking renumbers devices, so a
mask of `1` leaves the only visible GPU at `cuda:0` and pairing it with `cuda_index: 1` is always
wrong. The error message names both values.

### 5.3 Model backend

`typhoon-ai/typhoon-whisper-medium` — a Whisper-medium fine-tune (0.8B params) loaded through
`transformers`. An ordinary lockfile dependency: no source build, no pinned commit, no extras.

```python
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16)
ids = model.generate(feats, language="th", task="transcribe", max_new_tokens=440)
```

Two constraints shape the pipeline:

- **`language="th"`** (ISO-639-1), not `"th-TH"`.
- **A hard 30-second encoder window.** Longer clips are truncated by the feature extractor with
  no error, yielding partial text that reads like a bad transcription. `audio.max_duration_seconds`
  detects them; they are logged to the error journal and still transcribed (first window only),
  so no row is lost, and `validate` reports the count before GPU time is spent.

This replaced an earlier NeMo backend whose source build, pinned commit, optional extras and
prompt/tokenizer machinery produced six consecutive environment failures. See §12.

### 5.4 Model checkpoint

Auto-downloaded via `huggingface_hub` on first run, cached to a config-set path;
`local_model_path` overrides when set.

## 6. Architecture

```
src/thai_asr_batch/
├── config.py         # frozen dataclasses + load_config + resolve_device
├── logging_utils.py  # Bangkok-TZ formatter, session log file
├── records.py        # Record normal form + path<->segment_id rules
├── io_formats.py     # TSV/JSONL readers+writers behind one Protocol
├── audio.py          # probe + resample-on-the-fly (no-op fast path)
├── model.py          # Whisper wrapper, deferred import
├── batching.py       # length bucketing + failure isolation
├── checkpoint.py     # sharded resume store
├── inference.py      # orchestrator
├── combine.py        # streaming order-independent join
└── cli.py            # get_parser() + subcommands
```

### 6.1 Resume — sharded checkpoints

A flat 10M-line done-file re-scanned at every restart is too slow. Shard by `sid_idx0` (the
first `segment_id` component), which is also the first audio directory level, so checkpoint
locality matches disk locality.

```
checkpoints/<input-stem>/
├── manifest.json     # input path/size/mtime, format, config hash, schema version
├── shard-100.done    # newline-delimited segment_ids, append-only
└── shard-148.done
```

Plain append-only text, not SQLite: survives `kill -9` predictably and is inspectable with
`wc -l` mid-run. ~140 MB total at 10M ids; one shard's set is a few MB in RAM.

**Crash safety.** A kill mid-append leaves a truncated final line. On load, if a shard's last
line lacks a trailing `\n`, **discard it and truncate to the last complete newline**. A dropped
tail costs one free re-transcription; a *retained* truncated id would silently skip a real
segment forever. The same recovery applies to the predictions file before append-resume.

**Flush order is load-bearing.** fsync the *predictions* writer **before** the *checkpoint*.
Crashing between them re-transcribes a few segments (harmless, deduped on read). The reverse
order would mark segments done whose predictions were never durable — silent data loss.

**Cadence**: `flush_every_n` (default 200) or `flush_every_seconds` (default 30), whichever
comes first.

**Restart flow**: verify `manifest.json` against the input (hard stop on mismatch unless
`--force-resume`, since resuming against a changed input would interleave two corpora); stream
input; lazily load each shard's done-set on first encounter with LRU eviction. Because input is
naturally grouped by `sid_idx0`, each shard loads about once — effectively O(shard).

### 6.2 Batching and failure isolation

**Local bucketing, not a global sort.** A full sort would need the whole corpus in RAM and
destroy shard locality. Instead: buffer `sort_buffer_size` (2048) items, sort that window by
duration estimate, emit `batch_size` chunks, refill. A batch also closes early when
`max(duration) * len(batch)` exceeds `max_batch_duration_seconds`, bounding activation memory
when a window is all long clips.

**Two-tier isolation.** Attempt the batch whole; on exception re-run its members individually.
The bad clip fails alone with empty `pred_text` + an error record; its ~15 neighbours still
transcribe. Cost is one wasted batch pass per poison clip — negligible against an expected
~1000 bad clips at 10M scale. CUDA OOM is caught specially: log, `empty_cache()`, per-item retry.

**Dead-context guard.** After OOM the CUDA context can be unusable, making per-item retry also
fail and silently empty out every subsequent batch. Track consecutive whole-batch failures and
abort past `max_consecutive_batch_failures` (5). A job that stops loudly beats one that writes
empty rows for hours.

### 6.3 Audio handling

`ensure_conforming` is a no-op fast path: read the WAV header via the stdlib `wave` module
(cheap, no subprocess) and return the path unchanged when it already conforms. Only on mismatch
shell out to ffmpeg into a scratch file. This preserves the defensive guarantee without 10M
subprocess spawns, which would otherwise dominate runtime. `estimate_duration` prefers the
header, falls back to file-size arithmetic, and never raises — it feeds bucketing only, so
approximation is fine.

### 6.4 Model wrapper

`import transformers` happens **inside `load()`**, never at module import. This is what lets the
entire test suite and `--dry-run` run on a machine that never touches the model.

`load()` builds a `WhisperProcessor` and `WhisperForConditionalGeneration` from the configured
repo (or `local_model_path`), casts to the configured dtype and moves to the resolved device.
A native-library failure (`OSError`) is reported distinctly from a missing package, since a
partially extracted CUDA wheel presents that way.

`transcribe_batch` decodes each path to a mono float32 array via `audio.load_samples`, batches
them through the processor, calls `generate(language=..., task=..., max_new_tokens=...)` under
`torch.no_grad()`, and decodes with `skip_special_tokens=True`. It asserts length parity with
its input so a silent batch/result mismatch cannot corrupt the segment_id→text pairing.

### 6.5 Combine

Predictions are the smaller side, so build the hash from predictions and stream the input:
~2–3 GB for 10M rows, comfortable against 503 GB. `--low-memory` offers a sort + merge-join for
constrained hosts and should be the default if combine ever runs on the Mac.

Both text columns are NFKC-normalized on the way out (§4). Missing predictions are written with
empty `pred_text` and counted, keeping the combined file 1:1 with the input for WER scoring;
their ids go to `logs/missing-<stamp>.txt`. Extra predictions (present in predictions, absent
from input) are counted and warned — they signal a checkpoint/input mismatch.

### 6.6 CLI

| Command | Purpose |
|---|---|
| `transcribe` | Run ASR → `predicted-<name>.<ext>` |
| `combine` | Join input + predictions |
| `run` | `transcribe` then `combine` |
| `validate` | Config + audio-path sanity, model never loaded |
| `status` | Resume progress without attaching to the running process |

Flags: `--input`, `--config`, `--audio-root`, `--output-dir`, `--limit N`, `--dry-run`,
`--no-resume`, `--force-resume`, `--batch-size`, `--device`, `--log-level`, `--low-memory`.
CLI overrides YAML; config stays authoritative.

## 7. Configuration

All hyperparameters, paths, and settings live in `configs/*.yaml` — never hard-coded.
`default.yaml` holds the baseline; `vps_h100.yaml` overrides paths and raises `io_workers`;
`local_cpu.yaml` forces CPU with a tiny batch size for smoke tests. Key groups: `paths`,
`model`, `device`, `batch`, `audio`, `checkpoint`, `runtime`.

## 8. Conventions

- `src/` layout; small single-responsibility modules, one per pipeline stage.
- `typing` annotations on **all** variables and functions.
- Bracketed imports when pulling more than one object: `from bar import (foo, fizz)`.
- `pathlib.Path` throughout.
- `logging` to stdout **and** a session file; formatter includes asctime, levelname, name,
  lineno, message; date format `'%Y-%m-%d %H:%M:%S'` in **Asia/Bangkok** (enforced via a custom
  formatter, since the VPS is likely UTC).
- `logs/log-[datetime].txt` per session; failed clips also to `logs/errors-<stamp>.jsonl`.
- `get_parser()` builds the argparse parser.
- Dependencies pinned in `pyproject.toml` with a committed `uv.lock` (§9); `requirements.txt`
  survives only as a pointer stub.
- Bash script for environment setup and running.

## 9. Dependency management and setup

Dependencies use **uv in project mode**, per the project's `uv-python-project-setup` skill:
`pyproject.toml` declares them, `uv.lock` pins exact resolved versions, and both are committed.
`.python-version` pins the interpreter to 3.12, which removes the need to probe for a `python3`
on PATH (the local one is 3.9.6, not the 3.10.20 in SYSTEM.md).

Dependencies are added only with `uv add` / `uv add --dev` — never `uv pip install`, never by
hand-editing `pyproject.toml`, since both bypass the lockfile and break reproducibility.

Every dependency, the model backend included, resolves from `uv.lock`. There is no out-of-band
install step.

**`HF_HOME` is redirected** to `<repo>/.hf-cache` alongside `TMPDIR` and `UV_CACHE_DIR`: the
checkpoint is several GB and would otherwise land in `~/.cache`, potentially on a smaller
filesystem than the one deliberately chosen.


`setup_and_run.sh`:

1. Verify `uv` is present.
2. `uv lock --check` — abort if `pyproject.toml` and `uv.lock` have drifted.
3. `uv sync --frozen` — install exactly the committed lockfile, no re-resolution.
   If present, verify `git rev-parse HEAD` starts with `907edfd` and abort on drift — the single
   most likely cause of a mysterious `EncDecRNNTBPEModelWithPrompt` failure.
5. Export `CUDA_VISIBLE_DEVICES` and the CUDA wheel loader path.
4. Verify `ffmpeg -version`; import-check `torch` and `transformers`.
6. `exec uv run --no-sync python -m thai_asr_batch.cli "$@"`.

Flags `--setup-only` / `--skip-setup`, and `uv run --no-sync` on the exec, so a 10-day run never
stalls re-resolving dependencies on restart. Idempotent and safe to re-run.

## 10. Testing

Tests run in seconds on the Mac with **no GPU and transformers unused**, guaranteed by the deferred
import. `conftest.py` builds real tiny 16 kHz WAVs via the stdlib `wave` module (no binary
fixtures in git) and a `FakeAsrModel` returning deterministic text, raising on ids containing
`"boom"` to drive failure paths.

| Module | Priority coverage |
|---|---|
| `checkpoint` | Truncated-final-line recovery, shard reload, LRU eviction, manifest mismatch. **Highest value** — most likely to be wrong, most expensive to get wrong. |
| `io_formats` | Headerless round-trip; blank/tab-less/multi-tab/BOM/empty-text rows; `predicted-` naming; JSONL stem derivation; `detect_format` on misnamed files. |
| `records` | The `148-148801-11` path rule; malformed ids raise. |
| `batching` | Item conservation over random durations; size and duration caps. |
| `inference` | Resume skips completed; `--limit`; poison clip fails alone; consecutive-failure abort. |
| `combine` | Shuffled order still joins; **NFKC (สระอำ in both spellings compares equal)**; missing/extra accounting. |
| `cli`/`config` | `--dry-run` never constructs the model; MPS never chosen unless opted in. |

## 11. Execution ladder

1. `validate` on the full corpus — cheap, catches path/config errors before GPU time.
2. `transcribe --limit 100` locally on CPU.
3. `transcribe --limit 5000` on the H100 — measure real throughput, confirm the ~42x RTF estimate.
4. Full run under tmux/nohup, monitored via `status`.
5. `combine`.

## 12. Risk register

| Risk | Mitigation |
|---|---|
| Process killed (OOM-killer, reboot, SSH drop) | Sharded checkpoints + fsync cadence; `--skip-setup` restarts in seconds. |
| Partial line from `kill -9` | Truncate-to-last-newline recovery on checkpoint and predictions files. |
| GPU 1 accidentally used | `cuda_index: 0` in config **and** `CUDA_VISIBLE_DEVICES=0` in the runner. |
| CUDA OOM on a long-clip batch | `max_batch_duration_seconds` cap; OOM → `empty_cache()` → per-item retry. |
| Poison clip voids a batch | Two-tier isolation; empty `pred_text` + JSONL error record. |
| Unusable CUDA context after OOM | `max_consecutive_batch_failures` abort. |
| Memory growth over 10 days | LRU-bounded shard sets; generator readers; no hypothesis accumulation. |
| Disk fills | Startup size estimate; session-log rotation; `status` reports headroom. |
| CUDA `.so` files off the loader path | `export_cuda_library_path` globs `<site-packages>/nvidia/*/lib` into `LD_LIBRARY_PATH`, on the `--skip-setup` path too (it is per-process). |
| Clip longer than the 30s encoder window | `exceeds_window` flags it, logs to the error journal and transcribes the first window; `validate` reports the total up front. |
| Partial wheel from a small or full `/tmp` | `TMPDIR`/`UV_CACHE_DIR` default into the repo; `prepare_temp_dirs` aborts below `MIN_FREE_MB`; `verify_cuda_libraries` checks for `libcudnn.so.9` itself, since a truncated wheel still reports as installed. |
| torch importable but its libraries unloadable | `_cuda_available` catches `OSError` as well as `ImportError` and degrades to CPU with a warning, so `validate`/`--dry-run` still run on a broken box. |
| Silent quality regression | Sample predictions logged at DEBUG; `status` reports empty-prediction rate, so an all-empty run is caught in hour 1, not day 10. |

## 13. Validated against real data

The audio path rule and field names were open risks during design; ladder step 1 (`validate`)
resolved both against the real corpus.

**Field names** — a full scan parsed all **9,557,264** rows with zero failures, confirming the
headerless two-column shape.

**Path rule** — confirmed byte-for-byte against the VPS tree:

```
segment_id "8-8888-3"  ->  <audio_root>/train/8/8888/8-8888-3.wav
```

**`audio_root` is the PARENT of `train/`.** `records.py` appends its own `train` segment
(`AUDIO_SUBDIR`, `records.py:14`), so the corpus at
`/home/my/path/datasets/gigaspeech2/data/th/train/...` needs
`audio_root: /home/my/path/datasets/gigaspeech2/data/th`. Passing the `train/` directory itself
builds `.../th/train/train/8/...` and misses 100% of lookups — a trap worth naming, since it
reads as correct.

The first `validate` run missed all 9.5M clips for exactly this reason: `audio_root` still held
a value guessed during planning. It cost 123 seconds to find, rather than days of GPU time,
which is what that ladder step exists for.

### Still open

GPU-only paths — OOM recovery, `empty_cache()`, the `max_consecutive_batch_failures` abort — are
unit-tested with fakes but have never run on real hardware. Ladder step 3
(`transcribe --limit 5000`) is their first real exercise.
