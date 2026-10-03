// shaders/ops.comp (its software-E4M3 build, ops_compat) for the Direct3D adapters: the elementwise tensor
// operations of the NR graph, selected by MODE. One thread per eight consecutive channels of one pixel, whole
// 32-bit words in and out, every half rounding the bit-level roundF16.
#include "common.hlsl"

#ifndef SPEC_0
#define SPEC_0 0
#endif
static const uint MODE = SPEC_0;
static const uint MODE_F32_TO_F16 = 0u;        // input features -> f16 A operand
static const uint MODE_QUANTIZE = 1u;          // f16 -> E4M3
static const uint MODE_DOWNSAMPLE_FP8 = 2u;    // raw f16 2x box pool -> E4M3
static const uint MODE_POST_BLEND = 3u;        // 2x upsample E4 * scale + adapter E4 * scale -> f16 raw + E4M3
static const uint MODE_UPSAMPLE_RESIDUAL = 4u; // f16 projection (low res) + E4 skip * scale -> E4M3 (+ f16 raw)

cbuffer Push : register(b0) {
  uint count;        // output elements (multiple of 8)
  uint channels;     // multiple of 8
  uint inWidth;
  uint inHeight;
  uint outWidth;
  uint outHeight;
  uint auxOffsetA;   // half units in binding 4
  uint auxOffsetB;
  uint dual;         // also publish f16 raw output (binding 6)
};

RWByteAddressBuffer in32 : register(u0);    // f32, eight per 32 bytes
RWByteAddressBuffer in16 : register(u1);    // halves, eight per 16 bytes
RWByteAddressBuffer in8 : register(u2);     // E4M3 codes, eight per 8 bytes
RWByteAddressBuffer skip8 : register(u3);
RWByteAddressBuffer aux16 : register(u4);
RWByteAddressBuffer out8 : register(u5);
RWByteAddressBuffer out16 : register(u6);

// Eight scale halves as f32 (aux offsets are not 16-byte aligned in general).
void auxScales(uint offset, uint channel, out float s[8]) {
  for (uint i = 0u; i < 8u; ++i) {
    const uint index = offset + channel + i;
    s[i] = f16ToF32((aux16.Load((index >> 1u) << 2u) >> ((index & 1u) * 16u)) & 0xffffu);
  }
}

void unpack8(uint4 words, out float v[8]) {
  for (uint i = 0u; i < 4u; ++i) { v[2u * i] = f16ToF32(words[i] & 0xffffu); v[2u * i + 1u] = f16ToF32(words[i] >> 16u); }
}

void unpackE4x8(uint2 codes, out float v[8]) {
  for (uint i = 0u; i < 8u; ++i) v[i] = e4m3ToF32((codes[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu);
}

// f32 values already rounded to half -> packed halves (exact).
uint packHalves2(float a, float b) { return f16BitsExact(a) | (f16BitsExact(b) << 16u); }
uint4 packHalves8(float v[8]) {
  return uint4(packHalves2(v[0], v[1]), packHalves2(v[2], v[3]), packHalves2(v[4], v[5]), packHalves2(v[6], v[7]));
}

uint quantizePair(uint word) { return e4m3CodeFromF16Bits(word & 0xffffu) | (e4m3CodeFromF16Bits(word >> 16u) << 8u); }
uint2 quantize8(uint4 halves) {
  return uint2(quantizePair(halves.x) | (quantizePair(halves.y) << 16u), quantizePair(halves.z) | (quantizePair(halves.w) << 16u));
}

// A half add or scale as the GLSL's half arithmetic does it: one operation, one rounding to half (the f32 result of
// two halves keeps enough bits for that to be the single rounding).
float addHalves(float a, float b) { precise float sum = a + b; return roundF16(sum); }

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  const uint group = id.x + id.y * 65535u * 256u;   // 8-channel group
  [branch] if (group * 8u >= count) return;
  const uint groupsPerPixel = channels / 8u;
  const uint c = (group % groupsPerPixel) * 8u;
  const uint pixel = group / groupsPerPixel;
  if (MODE == MODE_F32_TO_F16) {
    float v[8];
    for (uint i = 0u; i < 8u; ++i) v[i] = roundF16(asfloat(in32.Load(group * 32u + i * 4u)));
    out16.Store4(group * 16u, packHalves8(v));
  } else if (MODE == MODE_QUANTIZE) {
    out8.Store2(group * 8u, quantize8(in16.Load4(group * 16u)));
  } else if (MODE == MODE_DOWNSAMPLE_FP8) {
    const uint ox = pixel % outWidth, oy = pixel / outWidth;
    const uint sx = ox * 2u, sy = oy * 2u;
    uint4 result = uint4(0u, 0u, 0u, 0u);
    [branch] if (sx + 1u < inWidth && sy + 1u < inHeight) {
      // The 2x2 pool: three half adds as (a+b)+(c+d), one half multiply by 0.25, then the E4M3 conversion.
      float p00[8], p10[8], p01[8], p11[8], value[8];
      unpack8(in16.Load4((((sy * inWidth + sx) * channels + c) / 8u) * 16u), p00);
      unpack8(in16.Load4((((sy * inWidth + sx + 1u) * channels + c) / 8u) * 16u), p10);
      unpack8(in16.Load4(((((sy + 1u) * inWidth + sx) * channels + c) / 8u) * 16u), p01);
      unpack8(in16.Load4(((((sy + 1u) * inWidth + sx + 1u) * channels + c) / 8u) * 16u), p11);
      for (uint i = 0u; i < 8u; ++i) {
        const float top = addHalves(p00[i], p10[i]);
        const float bottom = addHalves(p01[i], p11[i]);
        const float sum = addHalves(top, bottom);
        precise float quarter = sum * 0.25;
        value[i] = roundF16(quarter);
      }
      result = packHalves8(value);
    }
    out8.Store2(group * 8u, quantize8(result));
  } else if (MODE == MODE_POST_BLEND) {
    const uint ox = pixel % outWidth, oy = pixel / outWidth;
    const uint source = ((oy >> 1u) * inWidth + (ox >> 1u)) * channels + c;
    float up[8], adapter[8], scaleA[8], scaleB[8], raw[8];
    unpackE4x8(in8.Load2((source / 8u) * 8u), up);
    unpackE4x8(skip8.Load2(group * 8u), adapter);
    auxScales(auxOffsetA, c, scaleA);
    auxScales(auxOffsetB, c, scaleB);
    // learned_post_blend: the input product rounded to half, the adapter product joining the final add unrounded.
    for (uint i = 0u; i < 8u; ++i) {
      precise float inputProduct = up[i] * scaleA[i];
      precise float inputValue = roundF16(inputProduct);
      precise float adapterProduct = adapter[i] * scaleB[i];
      precise float value = inputValue + adapterProduct;
      raw[i] = roundF16(value);
    }
    const uint4 halves = packHalves8(raw);
    out16.Store4(group * 16u, halves);
    out8.Store2(group * 8u, quantize8(halves));
  } else if (MODE == MODE_UPSAMPLE_RESIDUAL) {
    const uint ox = pixel % outWidth, oy = pixel / outWidth;
    const uint source = ((oy >> 1u) * inWidth + (ox >> 1u)) * channels + c;
    float projected[8], skip[8], scale[8], value[8];
    unpack8(in16.Load4((source / 8u) * 16u), projected);
    unpackE4x8(skip8.Load2(group * 8u), skip);
    auxScales(auxOffsetA, c, scale);
    // residualAfterMatmul + scaleResidual: round_f16(value + residual * scale).
    for (uint i = 0u; i < 8u; ++i) {
      precise float product = skip[i] * scale[i];
      precise float sum = projected[i] + product;
      value[i] = roundF16(sum);
    }
    const uint4 halves = packHalves8(value);
    out8.Store2(group * 8u, quantize8(halves));
    if (dual != 0u) out16.Store4(group * 16u, halves);
  }
}
