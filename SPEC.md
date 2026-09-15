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
| R1 | Use model `typhoon-ai/typhoon-asr-streaming-nemotron-0.6b` (NeMo FastConformer-Transducer, RNN-T). |
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

Both `pred_text` and `orig_text` are normalized to **Unicode NFC**.

This is non-negotiable for Thai. สระอำ encodes either as U+0E33 or as นิคหิต + สระอา; the two
render identically but compare unequal byte-wise. Without NFC on both sides, correct predictions
would be scored as errors. Beyond NFC the model output is left raw — further normalization is
lossy and belongs in a separate, tunable scoring step.

## 5. Environment

### 5.1 Hardware targets

| | VPS (primary) | Local (dev) |
|---|---|---|
| OS | Linux, CUDA 13.0 | macOS, Apple M4 Pro |
| Compute | 2× H100 80GB — **index 0 only** (index 1 has 0 MiB free) | MPS/CPU |
| RAM | 503 GB | 24 GB |
| Python | 3.12 | **3.9.6 on PATH** (not 3.10.20) |

### 5.2 Device policy

`cuda → cpu`. MPS is selected **only** behind an explicit `allow_mps` opt-in, because NeMo on
Apple MPS tends to fail outright for RNN-T rather than degrade gracefully. The runner also
exports `CUDA_VISIBLE_DEVICES=0` so the index-0 constraint holds belt-and-braces.

### 5.3 Critical dependency constraint

Stock `nemo_toolkit` does **not** contain `EncDecRNNTBPEModelWithPrompt`, which this model
requires. NeMo must be built from source at pinned commit **`907edfd`**:

```bash
git clone https://github.com/NVIDIA/NeMo && cd NeMo && git checkout 907edfd && pip install -e .
export NEMO_ROOT=/path/to/NeMo
```

Not pip-resolvable, therefore deliberately **absent from `requirements.txt`** and handled by
`setup_and_run.sh`. The README documents the rationale so a future reader does not "fix" it by
adding a pip pin.

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
├── model.py          # NeMo wrapper, deferred import
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

`import nemo.collections.asr` happens **inside `load()`**, never at module import. This is what
lets the entire test suite and `--dry-run` work with NeMo uninstalled.

`load()` runs the model-card sequence — `restore_from(map_location=device)`, `eval()`,
`set_inference_prompt("th-TH")`, `decoding.set_strip_lang_tags(True)` — each of the latter three
guarded by `hasattr` so version drift fails loudly naming commit `907edfd` instead of with an
opaque `AttributeError` ten minutes in.

`transcribe_batch` normalizes NeMo's return shape (it has returned both `List[Hypothesis]` and
`List[List[Hypothesis]]` across versions) and asserts length parity with its input.

### 6.5 Combine

Predictions are the smaller side, so build the hash from predictions and stream the input:
~2–3 GB for 10M rows, comfortable against 503 GB. `--low-memory` offers a sort + merge-join for
constrained hosts and should be the default if combine ever runs on the Mac.

Missing predictions are written with empty `pred_text` and counted, keeping the combined file
1:1 with the input for WER scoring; their ids go to `logs/missing-<stamp>.txt`. Extra
predictions (present in predictions, absent from input) are counted and warned — they signal a
checkpoint/input mismatch.

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
- Pinned `requirements.txt`.
- Bash script for environment setup and running.

## 9. Setup script

`setup_and_run.sh`:

1. **Pin the interpreter** via `PYTHON_BIN` (default `python3.12`). Local `python3` is 3.9.6, so
   trusting it would silently build a broken venv.
2. Create/activate `.venv`; install `requirements.txt`.
3. If `$NEMO_ROOT` unset/absent: clone NeMo, `git checkout 907edfd`, `pip install -e .`.
   If present, verify `git rev-parse HEAD` starts with `907edfd` and abort on drift — this is
   the single most likely cause of a mysterious `EncDecRNNTBPEModelWithPrompt` failure.
4. Export `NEMO_ROOT` and `CUDA_VISIBLE_DEVICES=0`.
5. Verify `ffmpeg -version`; import-check `nemo.collections.asr`.
6. `exec python -m thai_asr_batch.cli "$@"`.

Flags `--setup-only` / `--skip-setup` (a 10-day run must not re-resolve pip on every restart).
Idempotent and safe to re-run.

## 10. Testing

Tests run in seconds on the Mac with **no GPU and NeMo uninstalled**, guaranteed by the deferred
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
| `combine` | Shuffled order still joins; **NFC (สระอำ in both encodings compares equal)**; missing/extra accounting. |
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
| NeMo commit drift | Setup asserts `907edfd`; `load()` `hasattr`-guards each card-specific API. |
| Silent quality regression | Sample predictions logged at DEBUG; `status` reports empty-prediction rate, so an all-empty run is caught in hour 1, not day 10. |

## 13. Open risk

The data and GPU live on the VPS; this workspace is empty, so **the audio path rule and field
names have not been validated against real files**. Ladder step 1 (`validate`) exists precisely
to catch that before GPU time is spent.
