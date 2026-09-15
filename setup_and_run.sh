#!/usr/bin/env bash
#
# Environment setup + runner for the Thai ASR batch-inference pipeline.
#
# Dependencies are managed with uv in project mode: pyproject.toml + uv.lock are
# the source of truth, and `uv sync --frozen` installs exactly what is locked.
#
# The ASR backend is typhoon-ai/typhoon-whisper-medium, an ordinary
# transformers model: it installs straight from the lockfile with no source
# build, no pinned commit and no optional extras.
#
# Usage:
#   ./setup_and_run.sh --setup-only
#   ./setup_and_run.sh transcribe --input data/train_refined.tsv --config configs/vps_h100.yaml
#   ./setup_and_run.sh --skip-setup transcribe --input ...   # fast restart
#
# Environment overrides:
#   SCRATCH_ROOT     base for TMPDIR and UV_CACHE_DIR (default: <repo>)
#   ASR_TMPDIR       override the wheel unpack dir (default: <repo>/.tmp; an
#                    inherited TMPDIR is deliberately ignored — see note below)
#   ASR_UV_CACHE_DIR override uv's download cache (default: <repo>/.uv-cache)
#   MIN_FREE_MB      refuse to install below this free space (default: 15000)
#   ASR_HF_HOME      HuggingFace cache location (default: <repo>/.hf-cache)

set -Eeuo pipefail

readonly REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Where package installs unpack wheels and cache downloads.
#
# The default /tmp is often a small partition (sometimes a RAM-backed tmpfs).
# Large wheels unpack there before being copied into the venv, so running out
# of space produces a PARTIAL install: some .so files present, the rest
# silently missing. That is exactly how nvidia-cudnn-cu13 ended up shipping 2
# of ~10 libraries, with libcudnn.so.9 among the casualties, and it surfaces
# much later as an unrelated-looking ImportError.
#
# Default both onto the repo's own disk, which is sized for bulk data. Override
# either if the repo lives on a small volume:
#   SCRATCH_ROOT=/data/big ./setup_and_run.sh --setup-only
# NOTE: TMPDIR is almost always already set (macOS presets it; many Linux
# shells inherit /tmp). So `${TMPDIR:-default}` would keep the inherited value
# and this whole guard would silently do nothing — which is precisely the
# failure it exists to prevent. Honour an explicit ASR_TMPDIR override instead,
# and otherwise take control of TMPDIR unconditionally.
# Default into the repo itself: /tmp is too small for the CUDA wheels, while
# the working directory has room.
SCRATCH_ROOT="${SCRATCH_ROOT:-${REPO_DIR}}"
TMPDIR="${ASR_TMPDIR:-${SCRATCH_ROOT}/.tmp}"
UV_CACHE_DIR="${ASR_UV_CACHE_DIR:-${SCRATCH_ROOT}/.uv-cache}"
# The model checkpoint (several GB) otherwise lands in ~/.cache, which may sit
# on a different and smaller filesystem than the one chosen above.
HF_HOME="${ASR_HF_HOME:-${SCRATCH_ROOT}/.hf-cache}"
# uv prefers its own cache over TMPDIR for extraction, so all three are set.
export TMPDIR UV_CACHE_DIR HF_HOME

# Refuse to install with less headroom than the CUDA wheel set needs.
readonly MIN_FREE_MB="${MIN_FREE_MB:-15000}"

log() { printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

load_dotenv() {
  # Secrets live in .env (gitignored); .env.example is the committed template.
  # `set -a` exports every assignment so child processes inherit them.
  local env_file="${REPO_DIR}/.env"
  [[ -f "$env_file" ]] || return 0
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
  # Never log the value itself.
  log ".env loaded${HF_TOKEN:+ (HF_TOKEN set)}"
}

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

free_mb() {
  # Portable-enough free space in MB for a path (POSIX df -P, 1024-blocks).
  df -Pk "$1" 2>/dev/null | awk 'NR==2 {print int($4/1024)}'
}

prepare_temp_dirs() {
  mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$HF_HOME" \
    || die "could not create TMPDIR / UV_CACHE_DIR / HF_HOME under ${SCRATCH_ROOT}"
  log "TMPDIR=${TMPDIR}"
  log "UV_CACHE_DIR=${UV_CACHE_DIR}"
  log "HF_HOME=${HF_HOME}"

  local avail
  avail="$(free_mb "$TMPDIR")"
  [[ -n "$avail" ]] || return 0

  if (( avail < MIN_FREE_MB )); then
    # Hard stop rather than a warning: a partial wheel install fails later, in
    # a place that looks nothing like a disk problem.
    die "only ${avail} MB free on ${TMPDIR}, need ~${MIN_FREE_MB} MB.
The CUDA wheels (torch + cuDNN + cuBLAS + NCCL) unpack here before install, and
running out of space yields a silently PARTIAL install.
Point somewhere larger: SCRATCH_ROOT=/path/with/space $0 --setup-only"
  fi
  log "${avail} MB free on ${TMPDIR}"
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
  uv run --no-sync python - <<'PY' || die "environment check failed; see above"
import sys

try:
    import torch  # noqa: F401
except OSError as exc:
    print(
        f"torch failed to load a shared library: {exc}\n"
        "Either the nvidia-* wheels are off the loader path, or a wheel was only\n"
        "partially extracted (a full disk during install does this).\n"
        "Re-run ./setup_and_run.sh --setup-only.",
        file=sys.stderr,
    )
    raise SystemExit(1)

try:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: F401
except Exception as exc:
    print(f"could not import transformers: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"torch {torch.__version__} + transformers OK (cuda={torch.cuda.is_available()})")
PY
}

main() {
  cd "$REPO_DIR"
  # Before everything: the token may be needed by the very first download.
  load_dotenv
  ensure_uv

  if [[ "$skip_setup" -eq 0 ]]; then
    # Must precede every install step: uv sync unpacks wheels through
    # TMPDIR/UV_CACHE_DIR, and the model download lands in HF_HOME.
    prepare_temp_dirs
    verify_lockfile
    sync_environment
    export_cuda_library_path
    verify_environment
  else
    # Still needed on a fast restart: the loader path is per-process, so a long
    # run resumed with --skip-setup would otherwise fail to find the CUDA libs.
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
