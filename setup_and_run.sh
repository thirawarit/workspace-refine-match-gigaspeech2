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
# cu13 matches the VPS (CUDA 13.0); use asr,cu12 or plain asr elsewhere.
NEMO_EXTRAS="${NEMO_EXTRAS:-asr,cu13}"

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
Fix: git -C '${NEMO_ROOT}' checkout ${NEMO_COMMIT} && uv pip install -e '${NEMO_ROOT}[${NEMO_EXTRAS}]'"
    fi
    log "NeMo present at pinned commit ${NEMO_COMMIT}"
    # Re-run even when the clone exists: an earlier bare `-e` install (without
    # the extra) leaves the venv importable but missing hydra/omegaconf.
    if ! uv run --no-sync python -c "import hydra, omegaconf, lightning" >/dev/null 2>&1; then
      log "NeMo present but its [asr] dependencies are missing; reinstalling"
      install_nemo
    fi
  else
    log "cloning NeMo into ${NEMO_ROOT} (this takes a while)"
    git clone --quiet "$NEMO_REPO" "$NEMO_ROOT"
    git -C "$NEMO_ROOT" checkout --quiet "$NEMO_COMMIT"
    install_nemo
  fi
  export NEMO_ROOT
}

install_nemo() {
  # The [asr] extra is REQUIRED, not optional. A bare `-e "$NEMO_ROOT"` installs
  # nemo-toolkit's base dependencies only, which omit hydra-core, omegaconf and
  # lightning — the import then dies with "No module named 'hydra'".
  #
  # At 907edfd these live in [project.optional-dependencies].asr in NeMo's own
  # pyproject.toml; that commit has no requirements/*.txt files at all.
  # Note: the `asr-only` extra does NOT include hydra — it must be `asr`.
  #
  # cu13 adds numba-cuda[cu13] + cuda-python>=13,<14 to match the VPS's CUDA
  # 13.0. Override for a CUDA 12 host or a CPU-only box:
  #   NEMO_EXTRAS=asr,cu12 ./setup_and_run.sh --setup-only
  #   NEMO_EXTRAS=asr      ./setup_and_run.sh --setup-only
  log "installing NeMo with extras [${NEMO_EXTRAS}] into the uv venv"
  uv pip install -e "${NEMO_ROOT}[${NEMO_EXTRAS}]"
}

export_cuda_library_path() {
  # The nvidia-* wheels ship their shared objects under
  #   <site-packages>/nvidia/<component>/lib/
  # but nothing adds those directories to the dynamic loader's search path. The
  # symptom is an import that dies with e.g.
  #   libcudnn.so.9: cannot open shared object file
  # even though `uv pip list` shows nvidia-cudnn-cu13 installed.
  #
  # Discover the directories at runtime rather than hardcoding versions, so this
  # keeps working when the wheels are upgraded.
  local site_packages
  site_packages="$(uv run --no-sync python -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)" || return 0
  [[ -d "${site_packages}/nvidia" ]] || return 0

  local cuda_libs=""
  local lib_dir
  while IFS= read -r lib_dir; do
    cuda_libs="${cuda_libs:+${cuda_libs}:}${lib_dir}"
  done < <(find "${site_packages}/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | sort)

  if [[ -n "$cuda_libs" ]]; then
    export LD_LIBRARY_PATH="${cuda_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    log "added $(tr ':' '\n' <<<"$cuda_libs" | wc -l | tr -d ' ') nvidia wheel lib dir(s) to LD_LIBRARY_PATH"
  fi
}

verify_environment() {
  command -v ffmpeg >/dev/null 2>&1 \
    || log "WARNING: ffmpeg not found; only needed for non-conforming audio"
  uv run --no-sync python - <<'PY' || die "NeMo import failed; see the message above"
import sys

try:
    import torch  # noqa: F401
except OSError as exc:
    # Native library failure, not a missing Python package.
    print(
        f"torch failed to load a shared library: {exc}\n"
        "The nvidia-* wheels are installed but their .so files are not on the\n"
        "loader path. setup_and_run.sh exports LD_LIBRARY_PATH for this; if you\n"
        "are invoking python directly, export it yourself:\n"
        "  export LD_LIBRARY_PATH=\"$(uv run --no-sync python -c "
        "'import sysconfig,glob,os;"
        "p=sysconfig.get_paths()[\\\"purelib\\\"];"
        "print(\\\":\\\".join(sorted(glob.glob(os.path.join(p,\\\"nvidia\\\",\\\"*\\\",\\\"lib\\\")))))')"
        "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"",
        file=sys.stderr,
    )
    raise SystemExit(1)

try:
    import nemo.collections.asr  # noqa: F401
except ModuleNotFoundError as exc:
    # Almost always a missing [asr] extra rather than a broken NeMo checkout.
    print(
        f"could not import nemo.collections.asr: {exc}\n"
        "This usually means NeMo was installed without its [asr] extra.\n"
        "Fix: uv pip install -e \"$NEMO_ROOT[$NEMO_EXTRAS]\"  (default extras: asr,cu13)",
        file=sys.stderr,
    )
    raise SystemExit(1)
except Exception as exc:
    print(f"could not import nemo.collections.asr: {exc}", file=sys.stderr)
    raise SystemExit(1)

from nemo.collections.asr.models import ASRModel  # noqa: F401
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
    export_cuda_library_path
    verify_environment
  else
    export NEMO_ROOT
    # Still needed on a fast restart: the loader path is per-process, so a
    # 10-day run resumed with --skip-setup would otherwise fail to find cuDNN.
    export_cuda_library_path
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
