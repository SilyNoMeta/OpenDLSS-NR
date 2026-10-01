// The tensor-core arithmetic of docs/numerics.md, in integer code, for kernels that have no FP8 MMA (the
// compatibility backend). Bit for bit the CPU reference's ref::adaFp8Fdpa16 / ref::adaF16Fdpa8 and the WebGPU
// port's ada_fp8_fdpa16 / ada_f16_fdpa8:
//
//   E = max(exp(accumulator), max over the nonzero pairs of exp(a) + exp(b))
//   sum = trunc(acc * 2^(F - E)) + sum_i trunc(a_i b_i * 2^(F - E))        F = 13 (E4M3) or 24 (f16)
//   result = round_f16(sum * 2^E)
//
// With value = significand * 2^(exponent - mantissa bits), every term is an integer significand product shifted by a
// small exponent difference, and every sum fits an int: E4M3 products are at most 15 * 15 << 7 (16 of them plus an
// accumulator below 2^14), f16 products at most 2^22 << 4 (8 of them plus an accumulator below 2^25).
#ifndef DLSS_EXACT_MMA_GLSL
#define DLSS_EXACT_MMA_GLSL

#include "common.glsl"

// trunc(value * 2^shift) for an integer value: toward zero, as the reference's std::trunc.
int shiftTowardZero(int value, int shift) {
  if (shift >= 0) return value << shift;
  if (shift <= -31) return 0;
  return value >= 0 ? (value >> -shift) : -((-value) >> -shift);
}

// An E4M3 code as value = significand * 2^(exponent - 3): significand 0..15 with the sign applied, exponent >= -6
// (subnormals share -6). Zero, and the NaN code (no weight or activation holds it), give significand 0.
void e4m3Split(uint code, out int significand, out int exponent) {
  uint e = (code >> 3u) & 15u, m = code & 7u;
  int magnitude = ((code & 0x7fu) == 0x7fu) ? 0 : int(m | (e != 0u ? 8u : 0u));
  significand = (code & 0x80u) != 0u ? -magnitude : magnitude;
  exponent = e != 0u ? int(e) - 7 : -6;
}

// A finite half as value = significand * 2^(exponent - 10): significand up to 11 bits with the sign applied,
// exponent >= -14 (subnormals share -14).
void f16Split(uint bits, out int significand, out int exponent) {
  uint e = (bits >> 10u) & 31u, m = bits & 0x3ffu;
  int magnitude = int(m | (e != 0u ? 0x400u : 0u));
  significand = (bits & 0x8000u) != 0u ? -magnitude : magnitude;
  exponent = e != 0u ? int(e) - 15 : -14;
}

// Code `index` (0..15) of sixteen E4M3 codes packed little-endian in a uvec4.
uint e4m3Code(uvec4 codes, uint index) { return (codes[index >> 2u] >> ((index & 3u) * 8u)) & 0xffu; }

// One F13 group: sixteen E4M3 products into an f16 accumulator (half of an m16n8k32 step).
float16_t adaFp8Fdpa16(uvec4 a, uvec4 b, float16_t accumulator) {
  const uint accumulatorBits = f16Bits(accumulator);
  if ((accumulatorBits & 0x7c00u) == 0x7c00u) return accumulator;   // an infinite or NaN C passes through
  int maximumExponent = -21;
  int accumulatorSignificand = 0, accumulatorExponent = 0;
  if ((accumulatorBits & 0x7fffu) != 0u) {
    f16Split(accumulatorBits, accumulatorSignificand, accumulatorExponent);
    maximumExponent = accumulatorExponent;
  }
  int products[16], exponents[16];
  for (uint i = 0u; i < 16u; ++i) {
    int sa, ea, sb, eb;
    e4m3Split(e4m3Code(a, i), sa, ea);
    e4m3Split(e4m3Code(b, i), sb, eb);
    products[i] = sa * sb;
    exponents[i] = ea + eb;
    if (products[i] != 0) maximumExponent = max(maximumExponent, exponents[i]);
  }
  // E4M3 significands carry 3 fractional bits each: a product is significands * 2^(e - 6), so its F13 term is
  // trunc(product * 2^(e - E + 7)); a half carries 10, so the accumulator's is trunc(s * 2^(e - E + 3)).
  int units = accumulatorSignificand != 0 ? shiftTowardZero(accumulatorSignificand, accumulatorExponent - maximumExponent + 3) : 0;
  for (uint i = 0u; i < 16u; ++i)
    if (products[i] != 0) units += shiftTowardZero(products[i], exponents[i] - maximumExponent + 7);
  return float16_t(roundF16(ldexp(float(units), maximumExponent - 13)));
}

// RNE shift right for the F24 publication (shift >= 1).
uint shiftRightEven(uint value, uint shift) {
  if (shift > 31u) return 0u;
  return shiftRne(value, shift);
}

// An exact signed F24 sum at 2^binaryExponent to a half, rounded to nearest even (ref fixedToF16).
float16_t fixedToF16(int fixedSum, int binaryExponent) {
  if (fixedSum == 0) return float16_t(0.0);
  const bool negative = fixedSum < 0;
  const uint magnitude = uint(negative ? -fixedSum : fixedSum);
  const uint msb = uint(findMSB(magnitude));
  int valueExponent = int(msb) + binaryExponent;
  uint halfBits = negative ? 0x8000u : 0u;
  if (valueExponent >= -14) {
    uint significand = msb > 10u ? shiftRightEven(magnitude, msb - 10u) : magnitude << (10u - msb);
    if (significand >= 2048u) { significand = 1024u; valueExponent += 1; }
    if (valueExponent >= 16) halfBits |= 0x7c00u;
    else halfBits |= (uint(valueExponent + 15) << 10u) | (significand - 1024u);
  } else {
    const int subnormalScale = binaryExponent + 24;
    const uint mantissa = subnormalScale >= 0 ? magnitude << uint(subnormalScale) : shiftRightEven(magnitude, uint(-subnormalScale));
    halfBits |= min(mantissa, 1024u);
  }
  return f16FromBits(halfBits);
}

// One F24 group: eight f16 products into an f16 accumulator (half of an m16n8k16 f16 step).
float16_t adaF16Fdpa8(f16vec4 a0, f16vec4 a1, f16vec4 b0, f16vec4 b1, float16_t accumulator) {
  int maximumExponent = -21;
  int accumulatorSignificand = 0, accumulatorExponent = 0;
  const uint accumulatorBits = f16Bits(accumulator);
  if ((accumulatorBits & 0x7fffu) != 0u) {
    f16Split(accumulatorBits, accumulatorSignificand, accumulatorExponent);
    maximumExponent = accumulatorExponent;
  }
  int products[8], exponents[8];
  for (uint i = 0u; i < 8u; ++i) {
    int sa, ea, sb, eb;
    f16Split(f16Bits(i < 4u ? a0[i] : a1[i - 4u]), sa, ea);
    f16Split(f16Bits(i < 4u ? b0[i] : b1[i - 4u]), sb, eb);
    products[i] = sa * sb;
    exponents[i] = ea + eb;
    if (products[i] != 0) maximumExponent = max(maximumExponent, exponents[i]);
  }
  // 10 fractional bits per significand: a product's F24 term is trunc(product * 2^(e - E + 4)), the accumulator's
  // trunc(s * 2^(e - E + 14)).
  int fixedSum = accumulatorSignificand != 0 ? shiftTowardZero(accumulatorSignificand, accumulatorExponent - maximumExponent + 14) : 0;
  for (uint i = 0u; i < 8u; ++i)
    if (products[i] != 0) fixedSum += shiftTowardZero(products[i], exponents[i] - maximumExponent + 4);
  return fixedToF16(fixedSum, maximumExponent - 24);
}

// The two approximations of the network (docs/numerics.md), correctly rounded to half in integer code. Native
// evaluates them as rcp.approx.ftz.f32 / rsqrt.approx.ftz.f32 and then cvt.rn.f16; for every half input that is the
// correctly rounded half (checked exhaustively on an Ampere GPU). GLSL's 1.0 / x and inversesqrt carry no such
// guarantee, and the NVIDIA compiler was seen to evaluate float16_t(inversesqrt(float(x))) at reduced precision
// (one ULP low on 554.5 in a window normalize), so the compatibility kernels do not rely on them.

// A nonzero finite half as value = significand * 2^(exponent - 25), significand in [1024, 2047] (subnormals
// normalised), exponent the biased half exponent (below 1 for a normalised subnormal).
void halfParts(uint bits, out uint significand, out int exponent) {
  uint e = (bits >> 10u) & 31u, m = bits & 0x3ffu;
  if (e != 0u) { significand = m | 0x400u; exponent = int(e); return; }
  uint shift = 10u - uint(findMSB(m));
  significand = m << shift;
  exponent = 1 - int(shift);
}

// (q + fraction) * 2^binaryExponent to half bits, round to nearest even: q a positive integer, `inexact` whether a
// fraction below q's last bit is nonzero.
uint roundToHalf(uint q, bool inexact, int binaryExponent) {
  int msb = findMSB(q);
  int valueExponent = msb + binaryExponent;
  int shift = valueExponent >= -14 ? msb - 10 : -24 - binaryExponent;
  uint kept;
  if (shift <= 0) kept = q << uint(-shift);
  else if (shift > 31) kept = 0u;
  else {
    kept = q >> uint(shift);
    uint dropped = q & ((1u << uint(shift)) - 1u), halfway = 1u << uint(shift - 1);
    if (dropped > halfway || (dropped == halfway && (inexact || (kept & 1u) != 0u))) kept += 1u;
  }
  if (valueExponent < -14) return min(kept, 0x400u);   // subnormal (0x400 is the smallest normal)
  if (kept == 2048u) { kept = 1024u; valueExponent += 1; }
  if (valueExponent >= 16) return 0x7c00u;
  return (uint(valueExponent + 15) << 10u) | (kept - 1024u);
}

// round_f16(1 / x): 1 / x = (2^31 / s + fraction) * 2^(-6 - exponent).
float16_t reciprocalF16(float16_t x) {
  uint bits = f16Bits(x), sign = bits & 0x8000u;
  if ((bits & 0x7fffu) == 0u) return f16FromBits(sign | 0x7c00u);
  if ((bits & 0x7c00u) == 0x7c00u) return f16FromBits((bits & 0x3ffu) != 0u ? 0x7e00u : sign);
  uint s; int e;
  halfParts(bits, s, e);
  return f16FromBits(sign | roundToHalf(0x80000000u / s, (0x80000000u % s) != 0u, -6 - e));
}

// sign(y^2 * s - 2^36) for y < 2^14, s < 2^12, exactly.
int compareSquareScaled(uint y, uint s) {
  uint high, low;
  umulExtended(y * y, s, high, low);
  if (high != 16u) return high > 16u ? 1 : -1;
  return low != 0u ? 1 : 0;
}

// round_f16(1 / sqrt(x)) for x >= 0: with x = s2 * 2^t, t even, s2 in [1024, 4095], 1 / sqrt(x) =
// (2^18 / sqrt(s2)) * 2^(-18 - t / 2); y = floor(2^18 / sqrt(s2)) from an f32 estimate, settled exactly.
float16_t rsqrtF16(float16_t x) {
  uint bits = f16Bits(x);
  if ((bits & 0x7fffu) == 0u) return f16FromBits(0x7c00u);   // a zero row's norm: +inf, as native
  if ((bits & 0x8000u) != 0u || (bits & 0x7c00u) == 0x7c00u)
    return f16FromBits(((bits & 0x8000u) != 0u || (bits & 0x3ffu) != 0u) ? 0x7e00u : 0u);
  uint s2; int t;
  halfParts(bits, s2, t);
  t -= 25;
  if ((t & 1) != 0) { s2 *= 2u; t -= 1; }
  uint y = uint(262144.0 / sqrt(float(s2)));
  for (int i = 0; i < 4 && compareSquareScaled(y, s2) > 0; ++i) y -= 1u;
  for (int i = 0; i < 4 && compareSquareScaled(y + 1u, s2) <= 0; ++i) y += 1u;
  return f16FromBits(roundToHalf(y, compareSquareScaled(y, s2) != 0, -18 - t / 2));
}

// f16 + f16 rounded to half (the partition sums): the f32 sum, then RNE to half. Rounding twice is harmless here,
// since f32 carries 24 >= 2 * 11 + 2 significand bits, so this is the half add's single rounding.
float16_t addHalf(float16_t a, float16_t b) { return float16_t(roundF16(float(a) + float(b))); }

#endif
