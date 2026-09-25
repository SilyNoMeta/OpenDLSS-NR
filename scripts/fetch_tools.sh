#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tools="$root/tools"
mkdir -p "$tools"

arch="$(uname -m)"
if [[ "$arch" != "x86_64" ]]; then
  echo "The prebuilt glslang package is available for x86_64 only (detected $arch)." >&2
  echo "Install glslangValidator, CMake, and Ninja through your distribution, then clone Vulkan-Headers and volk into tools/." >&2
  exit 1
fi

glslang_version="16.6.0"
headers_tag="v1.4.363"
volk_tag="vulkan-sdk-1.4.357.0"
cmake_version="3.31.12"
ninja_version="1.13.2"

if [[ ! -x "$tools/glslang/bin/glslangValidator" && ! -x "$tools/glslang/bin/glslang" ]]; then
  archive="$tools/glslang.tar.gz"
  url="https://github.com/KhronosGroup/glslang/releases/download/$glslang_version/glslang-$glslang_version-linux-x86_64-release.tar.gz"
  echo "downloading $url"
  curl -fL "$url" -o "$archive"
  mkdir -p "$tools/glslang"
  tar -xzf "$archive" -C "$tools/glslang"
  rm "$archive"
fi

if [[ ! -f "$tools/Vulkan-Headers/include/vulkan/vulkan.h" ]]; then
  git clone --depth 1 --branch "$headers_tag" https://github.com/KhronosGroup/Vulkan-Headers.git "$tools/Vulkan-Headers"
fi
if [[ ! -f "$tools/volk/volk.c" ]]; then
  git clone --depth 1 --branch "$volk_tag" https://github.com/zeux/volk.git "$tools/volk"
fi

if [[ ! -x "$tools/cmake/bin/cmake" ]]; then
  archive="$tools/cmake.tar.gz"
  url="https://github.com/Kitware/CMake/releases/download/v$cmake_version/cmake-$cmake_version-linux-x86_64.tar.gz"
  echo "downloading $url"
  curl -fL "$url" -o "$archive"
  mkdir -p "$tools/cmake"
  tar -xzf "$archive" -C "$tools/cmake" --strip-components=1
  rm "$archive"
fi

if [[ ! -x "$tools/ninja/ninja" ]]; then
  archive="$tools/ninja.zip"
  url="https://github.com/ninja-build/ninja/releases/download/v$ninja_version/ninja-linux.zip"
  echo "downloading $url"
  curl -fL "$url" -o "$archive"
  mkdir -p "$tools/ninja"
  python3 - "$archive" "$tools/ninja" <<'PY'
import sys
import zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2])
PY
  chmod +x "$tools/ninja/ninja"
  rm "$archive"
fi

if [[ "${1:-}" == "--npm" ]]; then
  mkdir -p "$tools/gltf"
  npm install --no-audit --no-fund --prefix "$tools/gltf" \
    "@gltf-transform/core@4" "@gltf-transform/extensions@4" "@gltf-transform/functions@4" \
    meshoptimizer draco3dgltf
fi

"$tools/cmake/bin/cmake" --version | head -1
echo "ninja $("$tools/ninja/ninja" --version)"
echo "tools ready in $tools"
