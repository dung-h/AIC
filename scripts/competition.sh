#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Resolve the runtime at the deployment boundary.  The old default pointed
# at one workstation's home directory, which made a source bundle fail on a
# new Linux server even when its own .venv was healthy.
if [[ -n "${HCMAI_PYTHON:-}" ]]; then
  python_bin="${HCMAI_PYTHON}"
elif [[ -x "/home/user1/hcmai-venv/bin/python" ]]; then
  python_bin="/home/user1/hcmai-venv/bin/python"
elif [[ -x "${repo_root}/.venv/bin/python" ]]; then
  python_bin="${repo_root}/.venv/bin/python"
else
  python_bin="$(command -v python3 || true)"
fi

if [[ ! -x "$python_bin" ]]; then
  echo "Competition Python is missing or not executable: $python_bin" >&2
  echo "Set HCMAI_PYTHON to a native-Linux virtualenv interpreter." >&2
  exit 2
fi
python_real="$(realpath "$python_bin")"
case "$python_real" in
  /mnt/*)
    echo "Refusing competition runtime from a Windows-mounted virtualenv: $python_real" >&2
    echo "Use a virtualenv under /home (or another native Linux filesystem)." >&2
    exit 3
    ;;
esac

export PYTHONPATH="$repo_root"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Keep the fast native-Linux model copy when it exists, but allow every
# deployment to override it explicitly.  The fallback under the project is
# what makes a self-contained server bundle work without /home/user1.
if [[ -z "${HCMAI_LOCAL_VLM_PATH:-}" ]]; then
  if [[ -d "/home/user1/runtime_models/Qwen2.5-VL-7B-Instruct" ]]; then
    export HCMAI_LOCAL_VLM_PATH="/home/user1/runtime_models/Qwen2.5-VL-7B-Instruct"
  elif [[ -d "${repo_root}/models/Qwen2.5-VL-7B-Instruct" ]]; then
    export HCMAI_LOCAL_VLM_PATH="${repo_root}/models/Qwen2.5-VL-7B-Instruct"
  fi
fi

command_name="${1:-}"
if [[ -z "$command_name" ]]; then
  echo "Usage: $0 preflight|run|bootstrap [arguments...]" >&2
  exit 2
fi
shift
cd "$repo_root"

case "$command_name" in
  preflight)
    exec "$python_bin" -m src.cli.competition_ready "$@"
    ;;
  run)
    exec "$python_bin" -m src.cli.competition_run "$@"
    ;;
  bootstrap)
    exec "$python_bin" scripts/runtime_data_bootstrap.py "$@"
    ;;
  *)
    echo "Unknown command: $command_name (expected preflight, run, or bootstrap)" >&2
    exit 2
    ;;
esac
