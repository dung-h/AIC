#!/usr/bin/env bash
set -euo pipefail

# Run the project venv without inheriting desktop/Windows Python injection.
# This wrapper is intentionally narrow: it does not alter packages, models,
# data, or indexes.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/scripts/runtime_env.sh"
hcmai_load_dotenv "${HCMAI_DOTENV:-${PROJECT_ROOT}/.env}"

unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
# Keep the repository importable while preventing inherited desktop paths.
export PYTHONPATH="${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.venv/bin:${PATH:-}"

exec "${PROJECT_ROOT}/.venv/bin/python" "$@"
