#!/usr/bin/env bash

set -Eeuo pipefail

REMOTE_HOST="${REMOTE_HOST:-draco-oci-login-01.draco-oci-iad.nvidia.com}"
REMOTE_USER="${REMOTE_USER:-zekail}"
REMOTE_HOME_OVERRIDE="${REMOTE_HOME_OVERRIDE:-}"
REMOTE_STORAGE_OVERRIDE="${REMOTE_STORAGE_OVERRIDE:-}"
REMOTE_RSYNC_OVERRIDE="${REMOTE_RSYNC_OVERRIDE:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOCAL_REPO="${LOCAL_REPO:-${DEFAULT_REPO}}"
LOCAL_STORAGE="${LOCAL_STORAGE:-${STORAGE:-}}"

DRY_RUN=0
ASSUME_YES=0
CHECKSUM=0
BW_LIMIT=""
REMOTE_CODE_NAME="${REMOTE_CODE_NAME:-FastWAM}"
COMPONENTS=(code data models checkpoints)

# Paths are relative to $STORAGE and retain the same layout on Draco.
DATA_ITEMS=(
  data/libero
  data/robotwin2.0
  data/text_embeds_cache/robotwin_clean_lerobotv3
  data/RoboTwin2.0-lerobotv3.0/_pooled_stats/clean_50tasks_fastwam_dataset_stats.json
)

MODEL_ITEMS=(
  models/fastwam
  models/diffsynth
)

CHECKPOINT_ITEMS=(
  runs/FastWAM
)

DEPENDENCY_ITEMS=(
  src/robotwin/curobo-v0.7.8
)

usage() {
  cat <<'EOF'
Usage: scripts/migrate_fastwam_to_draco.sh [options]

Migrate FastWAM to Draco while preserving the local HOME/STORAGE layout.
SSH asks for the account password interactively; this script never stores it.

Options:
  --only LIST              Comma-separated components:
                           code,data,models,checkpoints,dependencies
  --host HOST              Remote host (default: Draco OCI login host)
  --user USER              Remote user (default: zekail)
  --remote-home PATH       Override auto-detected remote $HOME
  --remote-storage PATH    Override auto-detected remote $STORAGE
  --remote-rsync PATH      Override the remote rsync executable
  --remote-code-name NAME  Directory under remote $HOME (default: FastWAM)
  --bwlimit RATE           Pass an rsync bandwidth limit, e.g. 500m
  --checksum               Compare file checksums instead of size/mtime
  --dry-run                Show what rsync would transfer without writing
  --yes                    Skip the final confirmation
  -h, --help               Show this help

Examples:
  scripts/migrate_fastwam_to_draco.sh --dry-run
  scripts/migrate_fastwam_to_draco.sh --only code,models,checkpoints
  scripts/migrate_fastwam_to_draco.sh --bwlimit 500m --yes

The operation is restartable. Interrupted files remain in .rsync-partial and
are reused on the next invocation. No destination files are deleted.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'warning: %s\n' "$*" >&2
}

require_arg() {
  [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires an argument"
}

while (($#)); do
  case "$1" in
    --only)
      require_arg "$@"
      IFS=',' read -r -a COMPONENTS <<<"$2"
      shift 2
      ;;
    --host)
      require_arg "$@"
      REMOTE_HOST="$2"
      shift 2
      ;;
    --user)
      require_arg "$@"
      REMOTE_USER="$2"
      shift 2
      ;;
    --remote-home)
      require_arg "$@"
      REMOTE_HOME_OVERRIDE="$2"
      shift 2
      ;;
    --remote-storage)
      require_arg "$@"
      REMOTE_STORAGE_OVERRIDE="$2"
      shift 2
      ;;
    --remote-rsync)
      require_arg "$@"
      REMOTE_RSYNC_OVERRIDE="$2"
      shift 2
      ;;
    --remote-code-name)
      require_arg "$@"
      REMOTE_CODE_NAME="$2"
      shift 2
      ;;
    --bwlimit)
      require_arg "$@"
      BW_LIMIT="$2"
      shift 2
      ;;
    --checksum)
      CHECKSUM=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

((${#COMPONENTS[@]} > 0)) || die "--only must select at least one component"
for component in "${COMPONENTS[@]}"; do
  case "$component" in
    code|data|models|checkpoints|dependencies) ;;
    *) die "unknown component in --only: $component" ;;
  esac
done

component_enabled() {
  local requested="$1"
  local component
  for component in "${COMPONENTS[@]}"; do
    [[ "$component" == "$requested" ]] && return 0
  done
  return 1
}

for command in ssh rsync find sort tail; do
  command -v "$command" >/dev/null 2>&1 || die "required command not found: $command"
done

[[ -n "$LOCAL_STORAGE" ]] || die "local STORAGE is not set; export STORAGE or LOCAL_STORAGE"
[[ -d "$LOCAL_REPO" ]] || die "local repository does not exist: $LOCAL_REPO"
[[ -d "$LOCAL_STORAGE" ]] || die "local storage does not exist: $LOCAL_STORAGE"

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
CONTROL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fastwam-rsync.XXXXXX")"
chmod 700 "$CONTROL_DIR"
CONTROL_PATH="${CONTROL_DIR}/control"
MASTER_OPEN=0

SSH_OPTIONS=(
  -o ControlMaster=auto
  -o ControlPersist=600
  -o "ControlPath=${CONTROL_PATH}"
  -o ServerAliveInterval=60
  -o ServerAliveCountMax=10
)

cleanup() {
  if ((MASTER_OPEN)); then
    ssh "${SSH_OPTIONS[@]}" -O exit "$REMOTE" >/dev/null 2>&1 || true
  fi
  rm -rf -- "$CONTROL_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Opening SSH control connection to %s ...\n' "$REMOTE"
printf 'SSH may request the password once. The password is not stored.\n'
ssh "${SSH_OPTIONS[@]}" -MNf "$REMOTE"
MASTER_OPEN=1

remote_exec() {
  ssh "${SSH_OPTIONS[@]}" "$REMOTE" "$@"
}

REMOTE_HOME="$REMOTE_HOME_OVERRIDE"
if [[ -z "$REMOTE_HOME" ]]; then
  REMOTE_HOME="$(remote_exec 'printenv HOME')"
fi

REMOTE_STORAGE="$REMOTE_STORAGE_OVERRIDE"
if [[ -z "$REMOTE_STORAGE" ]]; then
  REMOTE_STORAGE="$(remote_exec 'bash -l -c "printenv STORAGE"' 2>/dev/null || true)"
fi
if [[ -z "$REMOTE_STORAGE" ]]; then
  REMOTE_STORAGE="$(remote_exec 'bash -i -c "printenv STORAGE"' 2>/dev/null | tail -n 1 || true)"
fi

REMOTE_HOME="${REMOTE_HOME//$'\r'/}"
REMOTE_STORAGE="${REMOTE_STORAGE//$'\r'/}"
[[ "$REMOTE_HOME" == /* ]] || die "could not resolve an absolute remote HOME"
[[ "$REMOTE_STORAGE" == /* ]] || die \
  "could not resolve remote STORAGE; pass --remote-storage PATH"

REMOTE_REPO="${REMOTE_HOME}/${REMOTE_CODE_NAME}"

printf -v remote_storage_q '%q' "$REMOTE_STORAGE"
remote_exec "test -d ${remote_storage_q}" || \
  die "remote STORAGE does not exist: $REMOTE_STORAGE"

REMOTE_RSYNC="$REMOTE_RSYNC_OVERRIDE"
if [[ -z "$REMOTE_RSYNC" ]]; then
  REMOTE_RSYNC="$(remote_exec 'command -v rsync' 2>/dev/null || true)"
fi
if [[ -z "$REMOTE_RSYNC" ]]; then
  candidate="${REMOTE_STORAGE%/}/miniforge3/envs/rsync/bin/rsync"
  printf -v candidate_q '%q' "$candidate"
  if remote_exec "test -x ${candidate_q}"; then
    REMOTE_RSYNC="$candidate"
  fi
fi
[[ "$REMOTE_RSYNC" == /* ]] || die \
  "could not find remote rsync; pass --remote-rsync PATH"

RSYNC_RSH="ssh -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=${CONTROL_PATH} -o ServerAliveInterval=60 -o ServerAliveCountMax=10"
RSYNC_OPTIONS=(
  --archive
  --no-owner
  --no-group
  --human-readable
  --info=progress2,stats2
  --partial
  --partial-dir=.rsync-partial
  --timeout=600
  "--rsync-path=${REMOTE_RSYNC}"
)
((DRY_RUN)) && RSYNC_OPTIONS+=(--dry-run)
((CHECKSUM)) && RSYNC_OPTIONS+=(--checksum)
[[ -n "$BW_LIMIT" ]] && RSYNC_OPTIONS+=("--bwlimit=${BW_LIMIT}")

remote_mkdir() {
  ((DRY_RUN)) && return 0
  local path_q
  printf -v path_q '%q' "$1"
  remote_exec "mkdir -p -- ${path_q}"
}

sync_contents() {
  local source="$1"
  local destination="$2"
  shift 2

  [[ -d "$source" ]] || die "source directory does not exist: $source"
  remote_mkdir "$destination"
  printf '\n==> %s/\n    -> %s:%s/\n' "$source" "$REMOTE" "$destination"
  rsync "${RSYNC_OPTIONS[@]}" "$@" -e "$RSYNC_RSH" \
    "${source}/" "${REMOTE}:${destination}/"
}

sync_storage_item() {
  local relative="$1"
  local source="${LOCAL_STORAGE}/${relative}"
  local remote_parent="${REMOTE_STORAGE}/$(dirname -- "$relative")"
  local extra_options=()

  # The extracted benchmark dataset is sufficient; the split download archive
  # duplicates roughly 74 GB and is not needed on the destination.
  if [[ "$relative" == "data/robotwin2.0" ]]; then
    extra_options+=(--exclude='robotwin2.0.tar.gz.part-*')
  fi

  if [[ ! -e "$source" && ! -L "$source" ]]; then
    warn "skipping missing storage item: $source"
    return 0
  fi

  remote_mkdir "$remote_parent"
  printf '\n==> %s\n    -> %s:%s/\n' "$source" "$REMOTE" "$remote_parent"
  rsync "${RSYNC_OPTIONS[@]}" "${extra_options[@]}" -e "$RSYNC_RSH" \
    "$source" "${REMOTE}:${remote_parent}/"
}

print_items() {
  local label="$1"
  shift
  local item
  printf '  %s:\n' "$label"
  for item in "$@"; do
    printf '    %s/%s\n' "$LOCAL_STORAGE" "$item"
  done
}

printf '\nMigration plan\n'
printf '  remote:       %s\n' "$REMOTE"
printf '  code:         %s -> %s\n' "$LOCAL_REPO" "$REMOTE_REPO"
printf '  storage:      %s -> %s\n' "$LOCAL_STORAGE" "$REMOTE_STORAGE"
printf '  remote rsync: %s\n' "$REMOTE_RSYNC"
printf '  components:   %s\n' "${COMPONENTS[*]}"
((DRY_RUN)) && printf '  mode:         dry-run\n'
[[ -n "$BW_LIMIT" ]] && printf '  bwlimit:      %s\n' "$BW_LIMIT"

if component_enabled data; then
  print_items data "${DATA_ITEMS[@]}"
fi
if component_enabled models; then
  print_items models "${MODEL_ITEMS[@]}"
fi
if component_enabled checkpoints; then
  print_items checkpoints "${CHECKPOINT_ITEMS[@]}"
fi
if component_enabled dependencies; then
  print_items dependencies "${DEPENDENCY_ITEMS[@]}"
fi

printf '\nRemote filesystem:\n'
remote_exec "df -h -- ${remote_storage_q}" || true

if ((!ASSUME_YES && !DRY_RUN)); then
  [[ -r /dev/tty ]] || die "confirmation requires a terminal; rerun with --yes"
  read -r -p 'Start migration? [y/N] ' answer </dev/tty
  [[ "$answer" == y || "$answer" == Y || "$answer" == yes || "$answer" == YES ]] || {
    printf 'Migration cancelled.\n'
    exit 0
  }
fi

if component_enabled code; then
  sync_contents "$LOCAL_REPO" "$REMOTE_REPO" \
    --compress \
    --exclude='/.cache/' \
    --exclude='/.pytest_cache/' \
    --exclude='/.ruff_cache/' \
    --exclude='/.venv/' \
    --exclude='/data' \
    --exclude='/logs/' \
    --exclude='/outputs/' \
    --exclude='/third_party/RoboTwin/assets' \
    --exclude='/third_party/RoboTwin/envs/curobo' \
    --exclude='/third_party/RoboTwin/policy/fastwam_policy' \
    --exclude='__pycache__/' \
    --exclude='*.pyc'
fi

if component_enabled data; then
  for item in "${DATA_ITEMS[@]}"; do
    sync_storage_item "$item"
  done
fi

if component_enabled models; then
  for item in "${MODEL_ITEMS[@]}"; do
    sync_storage_item "$item"
  done
fi

if component_enabled checkpoints; then
  for item in "${CHECKPOINT_ITEMS[@]}"; do
    sync_storage_item "$item"
  done
fi

if component_enabled dependencies; then
  for item in "${DEPENDENCY_ITEMS[@]}"; do
    sync_storage_item "$item"
  done
fi

if component_enabled code && ((!DRY_RUN)); then
  printf '\n==> Rebuilding remote FastWAM symlinks\n'
  printf -v remote_repo_q '%q' "$REMOTE_REPO"
  remote_exec "bash -s -- ${remote_repo_q} ${remote_storage_q}" <<'REMOTE_SCRIPT'
set -Eeuo pipefail

repo="$1"
storage="$2"

robotwin_assets="$storage/data/robotwin_assets"
if [[ ! -d "$robotwin_assets" ]]; then
  robotwin_assets="$storage/data/RoboTwin2.0-assets"
fi
[[ -d "$robotwin_assets" ]] || {
  printf 'error: no RoboTwin assets directory found under %s/data\n' "$storage" >&2
  exit 1
}

curobo="$storage/src/robotwin/curobo"
if [[ ! -d "$curobo" ]]; then
  curobo="$storage/src/robotwin/curobo-v0.7.8"
fi
[[ -d "$curobo" ]] || {
  printf 'error: no cuRobo directory found under %s/src/robotwin\n' "$storage" >&2
  exit 1
}

replace_link() {
  local target="$1"
  local link="$2"

  mkdir -p -- "$(dirname -- "$link")"
  if [[ -e "$link" && ! -L "$link" ]]; then
    printf 'error: refusing to replace non-symlink path: %s\n' "$link" >&2
    exit 1
  fi
  ln -sfn -- "$target" "$link"
  printf '  %s -> %s\n' "$link" "$target"
}

replace_link "$storage/data" "$repo/data"
replace_link "$robotwin_assets" "$repo/third_party/RoboTwin/assets"
replace_link "$curobo" "$repo/third_party/RoboTwin/envs/curobo"
replace_link "$repo/experiments/robotwin/fastwam_policy" \
  "$repo/third_party/RoboTwin/policy/fastwam_policy"
REMOTE_SCRIPT
fi

printf '\nMigration completed successfully.\n'
printf 'Remote code:    %s:%s\n' "$REMOTE" "$REMOTE_REPO"
printf 'Remote storage: %s:%s\n' "$REMOTE" "$REMOTE_STORAGE"
