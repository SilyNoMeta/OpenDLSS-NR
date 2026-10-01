#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${GLSLANG:-}" ]]; then
  glslang="$GLSLANG"
elif [[ -x "$root/tools/glslang/bin/glslangValidator" ]]; then
  glslang="$root/tools/glslang/bin/glslangValidator"
elif [[ -x "$root/tools/glslang/bin/glslang" ]]; then
  glslang="$root/tools/glslang/bin/glslang"
elif command -v glslangValidator >/dev/null 2>&1; then
  glslang="$(command -v glslangValidator)"
elif command -v glslang >/dev/null 2>&1; then
  glslang="$(command -v glslang)"
else
  echo "glslang not found; run scripts/fetch_tools.sh or set GLSLANG." >&2
  exit 1
fi

python="${PYTHON:-python3}"
shader_out="$root/build/shaders"
ptx_out="$root/build/ptx"
mkdir -p "$shader_out" "$ptx_out"

for source in "$root"/shaders/*.comp; do
  "$glslang" -V --target-env vulkan1.3 -I"$root/shaders" "$source" \
    -o "$shader_out/$(basename "${source%.comp}").spv"
done
# ops.comp once more without the hardware E4M3 conversions (-DDLSS_SOFTWARE_E4M3), for the compatibility backend
for name in ops; do
  "$glslang" -V --target-env vulkan1.3 -DDLSS_SOFTWARE_E4M3 -I"$root/shaders" "$root/shaders/$name.comp" \
    -o "$shader_out/${name}_compat.spv"
done

for k in 64 128 256; do
  "$python" "$root/scripts/ptx/mlp_e4m3.py" "$k" "$ptx_out/mlp_e4m3_K$k.ptx" 3 1 1 >/dev/null
done
for c in 64 128 256 512; do
  "$python" "$root/scripts/ptx/qkv_e4m3.py" "$c" "$ptx_out/qkv_e4m3_K$c.ptx" 80 >/dev/null
done
for k in 32 64 128 256 512 1024; do
  for f in 5 13 4 8 6; do
    "$python" "$root/scripts/ptx/gemm2_e4m3.py" "$k" "$f" "$ptx_out/gemm2_e4m3_K${k}_f$f.ptx" >/dev/null
  done
done
while read -r c r x; do
  "$python" "$root/scripts/ptx/ffn_e4m3.py" "$c" "$r" "$ptx_out/ffn_e4m3_C${c}_R$r.ptx" "$x" >/dev/null
  "$python" "$root/scripts/ptx/ffn_e4m3.py" "$c" "$r" "$ptx_out/ffn_e4m3_C${c}_R${r}_proj.ptx" "$x" 1 >/dev/null
done <<'EOF'
64 4 0
128 4 0
256 3 80
256 4 64
EOF
while read -r k f p s; do
  "$python" "$root/scripts/ptx/gemmt_e4m3.py" "$k" "$f" "$p" "$s" "$ptx_out/gemmt_e4m3_K${k}_f${f}_p${p}_s$s.ptx" >/dev/null
done <<'EOF'
4096 1 32 4
1024 0 16 2
1024 1 8 4
1024 0 8 4
EOF
while read -r s f; do
  "$python" "$root/scripts/ptx/reduce_e4m3.py" "$s" "$f" "$ptx_out/reduce_e4m3_s${s}_f$f.ptx" >/dev/null
done <<'EOF'
4 4
2 8
4 8
EOF
while read -r k f s; do
  "$python" "$root/scripts/ptx/gemmv_e4m3.py" "$k" "$f" "$s" "$ptx_out/gemmv_e4m3_K${k}_f${f}_s$s.ptx" >/dev/null
  "$python" "$root/scripts/ptx/gemmv_e4m3.py" "$k" "$f" "$s" "$ptx_out/gemmv_e4m3_K${k}_f${f}_s${s}_m96.ptx" 4 6 0 0 0 96 >/dev/null
done <<'EOF'
1024 6 1
4096 5 4
1024 8 2
1024 5 4
EOF
for padded in 64 128 192 256; do
  "$python" "$root/scripts/ptx/global_attention_e4m3.py" "$padded" "$ptx_out/global_attention_e4m3_p$padded.ptx" >/dev/null
done
"$python" "$root/scripts/ptx/global_attention_stream_e4m3.py" normalize "$ptx_out/global_normalize_e4m3.ptx" >/dev/null
"$python" "$root/scripts/ptx/global_attention_stream_e4m3.py" attention "$ptx_out/global_attention_stream_e4m3.ptx" 4 >/dev/null
while read -r k f; do
  "$python" "$root/scripts/ptx/gemmv_e4m3.py" "$k" "$f" 1 "$ptx_out/gemmv_e4m3_K${k}_f${f}_s1_m96.ptx" 4 6 0 0 0 96 >/dev/null
done <<'EOF'
512 4
512 5
512 13
256 5
4096 5
1024 8
1024 5
EOF
for f in 2 66 74 130 48; do
  "$python" "$root/scripts/ptx/block32_e4m3.py" "$f" "$ptx_out/block32_e4m3_f$f.ptx" >/dev/null
done

echo "shaders -> $shader_out"
echo "PTX -> $ptx_out"
