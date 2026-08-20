#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
output="${1:-$repo_root/dist/hcmai-source-$(date +%Y%m%d-%H%M%S).tar.gz}"
output="$(realpath -m "$output")"

if [[ ! -d "$repo_root/.git" ]]; then
  echo "Git repository is not initialized: $repo_root" >&2
  exit 2
fi
case "$output" in
  "$repo_root"/dist/*) ;;
  *)
    echo "Refusing to write bundle outside $repo_root/dist: $output" >&2
    exit 3
    ;;
esac

mkdir -p "$(dirname "$output")"
cd "$repo_root"
git ls-files --cached --others --exclude-standard -z \
  | tar --null --files-from=- --create --gzip --file="$output"
sha256sum "$output" > "$output.sha256"

echo "$output"
echo "$output.sha256"
