// shaders/gemm_fp8_compat.comp for the Direct3D adapters: the exact software E4M3 GEMM, bit for bit.
//
// D = A(E4M3, rows x K) * B(E4M3, K x N), accumulated in f16 as a chain of k32 steps in K order, each step two F13
// groups of 16 products; residual seed, partition sums, SiLU and E4M3 / f16 publication as in the GLSL.
//
// The GLSL computes one output element per invocation and stores it as one byte or one half. Shader model 5 stores
// 32-bit words only, so here one invocation computes four consecutive columns of one row and stores their four
// codes as one word and their four halves as two: a workgroup still covers 64 columns of 4 rows, and the dispatch
// grid is the GLSL's (row groups in x, column groups in y, batches in z). Needs N, the output stride and the output
// column offset to be multiples of 4, which every GEMM of the network is (they are multiples of 16).
#include "common.hlsl"

#ifndef SPEC_0
#define SPEC_0 32
#endif
#ifndef SPEC_2
#define SPEC_2 0
#endif
#ifndef SPEC_3
#define SPEC_3 0
#endif
static const uint K = SPEC_0;
static const uint FLAGS = SPEC_2;
static const uint PARTITION = SPEC_3;
static const uint PARTITION_DIVISOR = SPEC_3 != 0 ? SPEC_3 : 1;   // the modulo below is only meaningful with a partition

static const uint F_RESIDUAL = 1u;
static const uint F_SCALE_RESIDUAL = 2u;
static const uint F_SILU = 8u;
static const uint F_QUANTIZE = 16u;
static const uint F_DUAL = 32u;
static const uint F_RESIDUAL_E4 = 64u;
static const uint F_BROADCAST_INPUT = 128u;

cbuffer Push : register(b0) {
  uint rows;
  uint N;
  uint Nmatrix;
  uint weightColumnOffset;
  uint inputStride;
  uint inputColumnBase;
  uint outputStride;
  uint outputColumnOffset;
  uint auxHalfOffset;
  uint batches;
};

RWByteAddressBuffer a4 : register(u0);            // E4M3 [rows][inputStride], 16-byte rows
RWByteAddressBuffer b4 : register(u1);            // E4M3 [batch][K/32][Nmatrix][32]
RWByteAddressBuffer outputF16 : register(u2);
RWByteAddressBuffer residualF16 : register(u3);
RWByteAddressBuffer auxF16 : register(u4);
RWByteAddressBuffer outputE4 : register(u5);
RWByteAddressBuffer residualE4 : register(u6);

uint loadHalf(RWByteAddressBuffer buffer, uint index) { return (buffer.Load((index >> 1u) << 2u) >> ((index & 1u) * 16u)) & 0xffffu; }
uint loadByte(RWByteAddressBuffer buffer, uint index) { return (buffer.Load(index & ~3u) >> ((index & 3u) * 8u)) & 0xffu; }

#if (SPEC_2 & 8)
// a * b + c rounded once to f32 (the GLSL's fma): exact in double for half operands, then one conversion.
float fmaOnce(float a, float b, float c) { return (float)((double)a * (double)b + (double)c); }

// common.glsl mpCubicSilu: each step one f32 operation rounded once to half.
float mpCubicSilu(float value) {
  float bounded = roundF16(clamp(value, -4.0, 4.0));
  float absolute = roundF16(abs(bounded));
  float inner = roundF16(fmaOnce(-0.055908203125, absolute, 0.447265625));
  float polynomial = roundF16(fmaOnce(bounded, inner, 0.89453125));
  precise float product = value * polynomial;
  return roundF16(product);
}
#endif

[numthreads(16, 4, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
  const uint column0 = group.y * 64u + local.x * 4u;
  const uint batch = group.z;
  const uint row = group.x * 4u + local.y;
  [branch] if (column0 >= N || row >= rows || batch >= batches) return;
  const uint outputIndex0 = row * outputStride + outputColumnOffset + batch * N + column0;

  uint acc[4], total[4];
  uint c;
  for (c = 0u; c < 4u; ++c) {
    acc[c] = 0u;
    total[c] = 0u;
    if ((FLAGS & F_RESIDUAL) != 0u) {
      float residual = (FLAGS & F_RESIDUAL_E4) != 0u ? e4m3ToF32(loadByte(residualE4, outputIndex0 + c))
                                                    : f16ToF32(loadHalf(residualF16, outputIndex0 + c));
      // two halves (or an E4M3 value and a half): the f32 product is exact, so this is the one rounding
      if ((FLAGS & F_SCALE_RESIDUAL) != 0u) {
        precise float scaled = residual * f16ToF32(loadHalf(auxF16, auxHalfOffset + column0 + c));
        residual = roundF16(scaled);
      }
      acc[c] = f16BitsExact(residual);
    }
  }
  const uint aBase = (row * inputStride + inputColumnBase + (((FLAGS & F_BROADCAST_INPUT) != 0u) ? 0u : batch * K)) / 16u;
  const uint bColumn0 = weightColumnOffset + column0;
  for (uint kb = 0u; kb < K; kb += 32u) {
    const uint4 a0 = a4.Load4((aBase + kb / 16u) * 16u), a1 = a4.Load4((aBase + kb / 16u + 1u) * 16u);
    const bool closes = PARTITION != 0u && (((kb + 32u) % PARTITION_DIVISOR) == 0u || kb + 32u >= K);
    for (c = 0u; c < 4u; ++c) {
      const uint bBase = ((batch * (K / 32u) + kb / 32u) * Nmatrix + bColumn0 + c) * 2u;
      uint value = adaFp8Fdpa16(a0, b4.Load4(bBase * 16u), acc[c]);
      value = adaFp8Fdpa16(a1, b4.Load4((bBase + 1u) * 16u), value);
      if (closes) {
        if (kb < PARTITION) total[c] = value;
        else total[c] = addHalf(total[c], value);
        value = 0u;
      }
      acc[c] = value;
    }
  }
  for (c = 0u; c < 4u; ++c) {
    if (PARTITION != 0u) acc[c] = total[c];
#if (SPEC_2 & 8)
    acc[c] = f16BitsExact(mpCubicSilu(f16ToF32(acc[c])));
#endif
  }
  if ((FLAGS & F_QUANTIZE) == 0u)
    outputF16.Store2(outputIndex0 * 2u, uint2(acc[0] | (acc[1] << 16u), acc[2] | (acc[3] << 16u)));
  if ((FLAGS & (F_QUANTIZE | F_DUAL)) != 0u)
    outputE4.Store(outputIndex0, e4m3CodeFromF16Bits(acc[0]) | (e4m3CodeFromF16Bits(acc[1]) << 8u) |
                                     (e4m3CodeFromF16Bits(acc[2]) << 16u) | (e4m3CodeFromF16Bits(acc[3]) << 24u));
}
