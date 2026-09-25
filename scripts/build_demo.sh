#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="Release"
[[ "${1:-}" == "--debug" ]] && config="Debug"
[[ $# -le 1 ]] || { echo "usage: scripts/build_demo.sh [--debug]" >&2; exit 2; }

if [[ -x "$root/tools/cmake/bin/cmake" ]]; then cmake="$root/tools/cmake/bin/cmake"; else cmake="${CMAKE:-cmake}"; fi
if [[ -x "$root/tools/ninja/ninja" ]]; then ninja="$root/tools/ninja/ninja"; else ninja="${NINJA:-ninja}"; fi
if [[ -n "${GLSLANG:-}" ]]; then glslang="$GLSLANG"
elif [[ -x "$root/tools/glslang/bin/glslangValidator" ]]; then glslang="$root/tools/glslang/bin/glslangValidator"
elif [[ -x "$root/tools/glslang/bin/glslang" ]]; then glslang="$root/tools/glslang/bin/glslang"
else glslang="${GLSLANG:-glslangValidator}"
fi

install="$root/third_party/filament-install"
matc="$install/bin/matc"
[[ -x "$matc" ]] || { echo "$matc not found; build Filament first." >&2; exit 1; }
data="$root/build/data"
mkdir -p "$data/shaders" "$data/materials"
for source in "$root"/demo/shaders/*.comp; do
  "$glslang" -V --target-env vulkan1.3 -I"$root/demo/shaders" "$source" \
    -o "$data/shaders/$(basename "$source").spv"
done
for source in "$root"/demo/materials/*.mat; do
  "$matc" -a vulkan -p desktop -o "$data/materials/$(basename "${source%.mat}").filamat" "$source"
done

"$cmake" -S "$root/demo" -B "$root/build/demo-build" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$ninja" -DCMAKE_BUILD_TYPE="$config"
"$cmake" --build "$root/build/demo-build" --parallel "${DLSS5_BUILD_JOBS:-$(nproc)}"
echo "built $root/build/demo/dlss5-demo"
