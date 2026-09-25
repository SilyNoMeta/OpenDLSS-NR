#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="$root/third_party/filament"
tag="v1.77.0"

if [[ ! -f "$dir/CMakeLists.txt" ]]; then
  git -c core.autocrlf=false clone --depth 1 --branch "$tag" https://github.com/google/filament.git "$dir"
  git -C "$dir" config core.autocrlf false
fi

patch="$root/third_party/filament.patch"
if git -C "$dir" apply --check "$patch" 2>/dev/null; then
  git -C "$dir" apply "$patch"
  echo "patch applied"
elif git -C "$dir" apply --check --reverse "$patch" 2>/dev/null; then
  echo "patch already applied"
else
  echo "third_party/filament.patch does not apply cleanly to $tag" >&2
  exit 1
fi
echo "Filament $tag ready at $dir"
