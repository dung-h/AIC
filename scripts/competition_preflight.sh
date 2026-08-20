#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/scripts/runtime_env.sh"
hcmai_load_dotenv "${HCMAI_DOTENV:-${PROJECT_ROOT}/.env}"

if [[ -n "${HCMAI_PYTHON:-}" ]]; then
  PYTHON_BIN="${HCMAI_PYTHON}"
elif [[ -x "/home/user1/hcmai-venv/bin/python" ]]; then
  PYTHON_BIN="/home/user1/hcmai-venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  printf '%s\n' '{"schema":"hcmai.competition_preflight.v1","ready":false,"exit_code":1,"blockers":["python_environment"],"message":"No executable Linux Python found"}'
  exit 1
fi

if [[ -z "${HCMAI_LOCAL_VLM_PATH:-}" ]]; then
  if [[ -d "${PROJECT_ROOT}/models/Qwen2.5-VL-7B-Instruct" ]]; then
    export HCMAI_LOCAL_VLM_PATH="${PROJECT_ROOT}/models/Qwen2.5-VL-7B-Instruct"
  elif [[ -d "/home/user1/runtime_models/Qwen2.5-VL-7B-Instruct" ]]; then
    export HCMAI_LOCAL_VLM_PATH="/home/user1/runtime_models/Qwen2.5-VL-7B-Instruct"
  fi
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1

preflight_args=()
run_args=()
separator_seen=0
for argument in "$@"; do
  if [[ "${argument}" == "--" && ${separator_seen} -eq 0 ]]; then
    separator_seen=1
    continue
  fi
  if [[ ${separator_seen} -eq 0 ]]; then
    preflight_args+=("${argument}")
  else
    run_args+=("${argument}")
  fi
done

cd -- "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m src.cli.competition_ready "${preflight_args[@]}"

if [[ ${#run_args[@]} -gt 0 ]]; then
  if [[ "${run_args[0]}" == "python" || "${run_args[0]}" == "python3" ]]; then
    run_args[0]="${PYTHON_BIN}"
  fi
  exec "${run_args[@]}"
fi
