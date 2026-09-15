#!/usr/bin/env bash
#
# Environment setup + runner for the Thai ASR batch-inference pipeline.
#
# NeMo is NOT pip-installable for this model: stock nemo_toolkit lacks
# EncDecRNNTBPEModelWithPrompt. It must be built from source at commit 907edfd,
# which is why this script exists rather than a plain `pip install -r`.
#
# Usage:
#   ./setup_and_run.sh --setup-only
#   ./setup_and_run.sh transcribe --input data/train_refined.tsv --config configs/vps_h100.yaml
#   ./setup_and_run.sh --skip-setup transcribe --input ...   # fast restart

set -Eeuo pipefail

readonly REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly NEMO_COMMIT="907edfd"
readonly NEMO_REPO="https://github.com/NVIDIA/NeMo"

# The local machine's `python3` is 3.9.6, which would silently build a broken
# venv. Pin the interpreter explicitly; override with PYTHON_BIN=... .
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
NEMO_ROOT="${NEMO_ROOT:-${REPO_DIR}/NeMo}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"

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

ensure_python() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 \
    || die "interpreter '$PYTHON_BIN' not found; set PYTHON_BIN=<python3.x>"
  log "using $("$PYTHON_BIN" --version 2>&1)"
}

ensure_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    log "creating venv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  log "installing pinned requirements"
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r "${REPO_DIR}/requirements.txt"
  python -m pip install --quiet -e "${REPO_DIR}"
}

ensure_nemo() {
  if [[ -d "${NEMO_ROOT}/.git" ]]; then
    local head
    head="$(git -C "$NEMO_ROOT" rev-parse HEAD)"
    if [[ "$head" != ${NEMO_COMMIT}* ]]; then
      die "NeMo at ${NEMO_ROOT} is at ${head:0:7}, expected ${NEMO_COMMIT}.
This is the most likely cause of a missing EncDecRNNTBPEModelWithPrompt.
Fix: git -C '${NEMO_ROOT}' checkout ${NEMO_COMMIT} && pip install -e '${NEMO_ROOT}'"
    fi
    log "NeMo present at pinned commit ${NEMO_COMMIT}"
  else
    log "cloning NeMo into ${NEMO_ROOT} (this takes a while)"
    git clone --quiet "$NEMO_REPO" "$NEMO_ROOT"
    git -C "$NEMO_ROOT" checkout --quiet "$NEMO_COMMIT"
    log "installing NeMo (editable)"
    python -m pip install --quiet -e "$NEMO_ROOT"
  fi
  export NEMO_ROOT
}

verify_environment() {
  command -v ffmpeg >/dev/null 2>&1 \
    || log "WARNING: ffmpeg not found; only needed for non-conforming audio"
  python - <<'PY' || die "NeMo import failed; see the message above"
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

  if [[ "$skip_setup" -eq 0 ]]; then
    ensure_python
    ensure_venv
    ensure_nemo
    verify_environment
  else
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
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
  log "running: python -m thai_asr_batch.cli ${args[*]}"
  exec python -m thai_asr_batch.cli "${args[@]}"
}

main "$@"
