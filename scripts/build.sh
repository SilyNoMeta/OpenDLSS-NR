#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="Release"
build_shaders=1
for arg in "$@"; do
  case "$arg" in
    --debug) config="Debug" ;;
    --no-shaders) build_shaders=0 ;;
    *) echo "usage: scripts/build.sh [--debug] [--no-shaders]" >&2; exit 2 ;;
  esac
done

if [[ -x "$root/tools/cmake/bin/cmake" ]]; then cmake="$root/tools/cmake/bin/cmake"
elif command -v cmake >/dev/null 2>&1; then cmake="$(command -v cmake)"
else echo "CMake not found; run scripts/fetch_tools.sh." >&2; exit 1
fi
if [[ -x "$root/tools/ninja/ninja" ]]; then ninja="$root/tools/ninja/ninja"
elif command -v ninja >/dev/null 2>&1; then ninja="$(command -v ninja)"
else echo "Ninja not found; run scripts/fetch_tools.sh." >&2; exit 1
fi

if (( build_shaders )); then "$root/scripts/build_shaders.sh"; fi

"$cmake" -S "$root" -B "$root/build/linux" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$ninja" -DCMAKE_BUILD_TYPE="$config"
"$cmake" --build "$root/build/linux" --parallel "${DLSS5_BUILD_JOBS:-$(nproc)}"
echo "built $root/build/dlss5vk"
