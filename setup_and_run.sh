#!/usr/bin/env bash
#
# Environment setup + runner for the Thai ASR batch-inference pipeline.
#
# Dependencies are managed with uv in project mode: pyproject.toml + uv.lock are
# the source of truth, and `uv sync --frozen` installs exactly what is locked.
#
# NeMo is the one exception. It is NOT pip/uv-installable for this model: stock
# nemo_toolkit lacks EncDecRNNTBPEModelWithPrompt, so it must be built from
# source at commit 907edfd and is therefore installed into the uv venv
# separately, after the sync.
#
# Usage:
#   ./setup_and_run.sh --setup-only
#   ./setup_and_run.sh transcribe --input data/train_refined.tsv --config configs/vps_h100.yaml
#   ./setup_and_run.sh --skip-setup transcribe --input ...   # fast restart

set -Eeuo pipefail

readonly REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly NEMO_COMMIT="907edfd"
readonly NEMO_REPO="https://github.com/NVIDIA/NeMo"

NEMO_ROOT="${NEMO_ROOT:-${REPO_DIR}/NeMo}"

log() { printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

setup_only=0
skip_setup=0
args=()
for arg in "$@"; do
  case "$arg" in
    --setup-only) setup_only=1 ;;
    --skip-setup) skip_setup=1 ;;
    *) args+=("$arg") ;;
  esac
done

ensure_uv() {
  command -v uv >/dev/null 2>&1 || die "uv not found. Install it:
  curl -LsSf https://astral.sh/uv/install.sh | sh"
  log "using uv $(uv --version)"
}

sync_environment() {
  # --frozen installs exactly the committed lockfile with no re-resolution.
  # The Python version comes from .python-version, so the interpreter is
  # consistent across machines without probing for a `python3` on PATH.
  log "syncing locked dependencies"
  uv sync --frozen
}

verify_lockfile() {
  # Fails if pyproject.toml and uv.lock have drifted apart.
  uv lock --check >/dev/null 2>&1 \
    || die "uv.lock is out of date with pyproject.toml. Run: uv lock"
  log "lockfile is current"
}

ensure_nemo() {
  if [[ -d "${NEMO_ROOT}/.git" ]]; then
    local head
    head="$(git -C "$NEMO_ROOT" rev-parse HEAD)"
    if [[ "$head" != ${NEMO_COMMIT}* ]]; then
      die "NeMo at ${NEMO_ROOT} is at ${head:0:7}, expected ${NEMO_COMMIT}.
This is the most likely cause of a missing EncDecRNNTBPEModelWithPrompt.
Fix: git -C '${NEMO_ROOT}' checkout ${NEMO_COMMIT} && uv pip install -e '${NEMO_ROOT}'"
    fi
    log "NeMo present at pinned commit ${NEMO_COMMIT}"
  else
    log "cloning NeMo into ${NEMO_ROOT} (this takes a while)"
    git clone --quiet "$NEMO_REPO" "$NEMO_ROOT"
    git -C "$NEMO_ROOT" checkout --quiet "$NEMO_COMMIT"
    log "installing NeMo from source into the uv venv"
    # The one place `uv pip` is correct: NeMo cannot be expressed in uv.lock,
    # so it is installed into the synced venv rather than declared as a dep.
    uv pip install -e "$NEMO_ROOT"
  fi
  export NEMO_ROOT
}

verify_environment() {
  command -v ffmpeg >/dev/null 2>&1 \
    || log "WARNING: ffmpeg not found; only needed for non-conforming audio"
  uv run python - <<'PY' || die "NeMo import failed; see the message above"
import sys
try:
    import nemo.collections.asr  # noqa: F401
except Exception as exc:
    print(f"could not import nemo.collections.asr: {exc}", file=sys.stderr)
    raise SystemExit(1)
print("nemo.collections.asr OK")
PY
}

main() {
  cd "$REPO_DIR"
  ensure_uv

  if [[ "$skip_setup" -eq 0 ]]; then
    verify_lockfile
    sync_environment
    ensure_nemo
    verify_environment
  else
    export NEMO_ROOT
    log "skipping setup (--skip-setup)"
  fi

  # GPU index 1 is fully occupied on the VPS; belt-and-braces with the config.
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

  if [[ "$setup_only" -eq 1 ]]; then
    log "setup complete"
    exit 0
  fi

  [[ ${#args[@]} -gt 0 ]] || die "no command given; try: $0 --help"
  log "running: uv run thai-asr-batch ${args[*]}"
  # --no-sync: the sync already happened above (or was deliberately skipped),
  # so a long run never stalls re-resolving dependencies on restart.
  exec uv run --no-sync python -m thai_asr_batch.cli "${args[@]}"
}

main "$@"
