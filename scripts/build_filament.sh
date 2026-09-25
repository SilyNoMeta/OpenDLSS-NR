#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="Release"
[[ "${1:-}" == "--debug" ]] && config="Debug"
[[ $# -le 1 ]] || { echo "usage: scripts/build_filament.sh [--debug]" >&2; exit 2; }

command -v clang >/dev/null 2>&1 && command -v clang++ >/dev/null 2>&1 || {
  echo "Filament's Linux desktop build requires clang and clang++." >&2
  exit 1
}

if [[ -x "$root/tools/cmake/bin/cmake" ]]; then cmake="$root/tools/cmake/bin/cmake"; else cmake="${CMAKE:-cmake}"; fi
if [[ -x "$root/tools/ninja/ninja" ]]; then ninja="$root/tools/ninja/ninja"; else ninja="${NINJA:-ninja}"; fi
src="$root/third_party/filament"
out="${DLSS5_FILAMENT_BUILD_DIR:-$root/build/filament-${config,,}}"
install="$root/third_party/filament-install"

[[ -f "$src/CMakeLists.txt" ]] || { echo "run scripts/fetch_filament.sh first" >&2; exit 1; }
"$cmake" -S "$src" -B "$out" -G Ninja -DCMAKE_MAKE_PROGRAM="$ninja" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE="$config" -DCMAKE_INSTALL_PREFIX="$install" \
  -DFILAMENT_SUPPORTS_VULKAN=ON -DFILAMENT_SUPPORTS_OPENGL=OFF \
  -DFILAMENT_SKIP_SAMPLES=ON -DFILAMENT_ENABLE_MATDBG=OFF -DFILAMENT_ENABLE_FGVIEWER=OFF
"$cmake" --build "$out" --target install --parallel "${DLSS5_BUILD_JOBS:-$(nproc)}"
echo "Filament installed to $install"
