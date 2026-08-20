#!/usr/bin/env bash
# Shared, non-evaluating dotenv loader for HCMAI operational wrappers.
#
# It accepts only KEY=VALUE / export KEY=VALUE lines.  It deliberately does
# not `source` .env: secrets are configuration, not executable shell code.
# Exported process values always win over values in the file.

hcmai_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

hcmai_apply_legacy_env_aliases() {
  local legacy canonical value
  while IFS=':' read -r legacy canonical; do
    value="${!legacy:-}"
    if [[ -z "${!canonical+x}" && -n "$value" ]]; then
      export "$canonical=$value"
    fi
  done <<'EOF'
DO_INFERENCE_BASE:VLM_BASE_URL
DO_INFERENCE_KEY:VLM_API_KEY
DO_VLM_MODEL:VLM_MODEL
EOF
}

hcmai_load_dotenv() {
  local env_file="${1:?dotenv path is required}"
  local raw line key value

  # Treat pre-existing legacy names as explicit process overrides before file
  # values are materialized, matching src.utils.paths.load_runtime_env().
  hcmai_apply_legacy_env_aliases
  [[ -f "$env_file" ]] || return 0

  while IFS= read -r raw || [[ -n "$raw" ]]; do
    line="$(hcmai_trim "$raw")"
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    key="$(hcmai_trim "${line%%=*}")"
    value="$(hcmai_trim "${line#*=}")"
    if [[ "$key" == "export "* ]]; then
      key="$(hcmai_trim "${key#export }")"
    fi
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ ${#value} -ge 2 ]] && {
      [[ "${value:0:1}" == "\"" && "${value: -1}" == "\"" ]] ||
      [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]];
    }; then
      value="${value:1:${#value}-2}"
    fi
    [[ -v "$key" ]] || export "$key=$value"
  done < "$env_file"

  # Map a legacy field contained in .env only when no canonical field was
  # already provided by the process or the same dotenv file.
  hcmai_apply_legacy_env_aliases
}
