// The numeric helpers of shaders/common.glsl and shaders/exact_mma.glsl for the Direct3D adapters (cs_5_0, so the
// same bytecode runs on Direct3D 11 and 12). Same publication points, same integer code, bit for bit.
//
// Shader model 5 has no 16-bit or 8-bit type and no narrow store. Halves and E4M3 codes are therefore carried as
// bit patterns in 32-bit words, every conversion is spelled on the bits (nothing is left to f32tof16 / f16tof32 or
// to an approximate intrinsic), and every kernel stores whole 32-bit words: one invocation owns every element of
// the words it writes, so no two invocations ever touch the same word.
//
// Bounds: a Direct3D 12 root descriptor has none, and the compiler may evaluate both sides of a flattened
// conditional. Every kernel therefore guards its range with a real branch ([branch]) and never forms an address
// outside its buffers, even on the side that is not taken.
#ifndef DLSS_COMMON_HLSL
#define DLSS_COMMON_HLSL

// 2^exponent for -126 <= exponent <= 127, exactly (ldexp is exp2-based and approximate in HLSL).
float powerOfTwo(int exponent) { return asfloat(uint(exponent + 127) << 23); }

// f32 -> RNE f16 -> f32, on the bit pattern (common.glsl roundF16).
float roundF16(float value) {
  uint bits = asuint(value);
  uint sign = bits & 0x80000000u;
  uint magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x7f800000u) return value;                           // inf / nan
  if (magnitude >= 0x477ff000u) return asfloat(sign | 0x7f800000u);     // >= 65520 overflows to inf
  if (magnitude < 0x38800000u) {                                        // below 2^-14: half subnormal grid 2^-24
    // round() is round-to-nearest-even (round_ne); the scaled value is below 2^10, so it is exact in f32.
    float scaled = round(asfloat(magnitude) * 16777216.0);
    // The GLSL adds a signed zero to keep the sign of a zero result; setting the sign bit is the same value.
    return asfloat(sign | asuint(scaled * 0.000000059604644775390625));
  }
  uint lsb = (magnitude >> 13u) & 1u;
  uint rounded = (magnitude + 0xfffu + lsb) & ~0x1fffu;
  return asfloat(sign | rounded);
}

// A half bit pattern to its f32 value, exactly.
float f16ToF32(uint bits) {
  uint sign = (bits & 0x8000u) << 16u;
  uint exponent = (bits >> 10u) & 31u, mantissa = bits & 0x3ffu;
  if (exponent == 31u) return asfloat(sign | 0x7f800000u | (mantissa << 13u));
  if (exponent == 0u) {
    float magnitude = float(mantissa) * 0.000000059604644775390625;   // mantissa * 2^-24, exact
    return asfloat(sign | asuint(magnitude));
  }
  return asfloat(sign | ((exponent + 112u) << 23u) | (mantissa << 13u));
}

// An f32 that already lies on the half grid (a roundF16 result) to its half bit pattern, exactly.
uint f16BitsExact(float value) {
  uint bits = asuint(value);
  uint sign = (bits >> 16u) & 0x8000u;
  uint magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x7f800000u) return sign | 0x7c00u | ((magnitude & 0x7fffffu) != 0u ? 0x200u : 0u);
  if (magnitude < 0x38800000u) return sign | uint(asfloat(magnitude) * 16777216.0);   // subnormal: an integer below 1024
  return sign | (((magnitude >> 23u) - 112u) << 10u) | ((magnitude >> 13u) & 0x3ffu);
}

uint shiftRne(uint value, uint shift) {
  uint quotient = value >> shift;
  uint remainderMask = (1u << shift) - 1u;
  uint remainder = value & remainderMask;
  uint halfway = 1u << (shift - 1u);
  return quotient + ((remainder > halfway || (remainder == halfway && (quotient & 1u) != 0u)) ? 1u : 0u);
}

// An already half-rounded value to its E4M3FN code: RNE, finite saturation at 448, NaN -> +0, signed zero kept.
uint e4m3CodeFromF16Bits(uint halfBits) {
  bool isNan = (halfBits & 0x7c00u) == 0x7c00u && (halfBits & 0x03ffu) != 0u;
  if (isNan) return 0u;
  if ((halfBits & 0x7fffu) == 0u) return (halfBits >> 8u) & 0x80u;
  uint negative = (halfBits & 0x8000u) != 0u ? 0x80u : 0u;
  uint exponent = (halfBits >> 10u) & 0x1fu;
  uint mantissa = halfBits & 0x03ffu;
  uint code;
  if (exponent == 31u) {
    code = 0x7eu;                     // inf saturates
  } else if (exponent <= 8u) {        // E4M3 subnormal
    uint significand = exponent == 0u ? mantissa : 1024u + mantissa;
    uint shift = exponent == 0u ? 15u : 16u - exponent;
    code = min(shiftRne(significand, shift), 8u);
  } else {
    uint e4Exponent = exponent - 8u;
    uint e4Mantissa = shiftRne(mantissa, 7u);
    if (e4Mantissa == 8u) { e4Mantissa = 0u; e4Exponent += 1u; }
    code = (e4Exponent > 15u || (e4Exponent == 15u && e4Mantissa > 6u)) ? 0x7eu : ((e4Exponent << 3u) | e4Mantissa);
  }
  return negative | code;
}

float e4m3ToF32(uint bits) {
  bool negative = (bits & 0x80u) != 0u;
  uint exponent = (bits >> 3u) & 0x0fu;
  uint mantissa = bits & 0x07u;
  float value;
  if (exponent == 0u) value = float(mantissa) * 0.001953125;
  else if (exponent == 15u && mantissa == 7u) value = 0.0;
  else value = (1.0 + float(mantissa) * 0.125) * asfloat((exponent + 120u) << 23u);
  return negative ? -value : value;
}

// trunc(value * 2^shift) for an integer value: toward zero.
int shiftTowardZero(int value, int shift) {
  if (shift >= 0) return value << shift;
  if (shift <= -31) return 0;
  return value >= 0 ? (value >> -shift) : -((-value) >> -shift);
}

// An E4M3 code as value = significand * 2^(exponent - 3).
void e4m3Split(uint code, out int significand, out int exponent) {
  uint e = (code >> 3u) & 15u, m = code & 7u;
  int magnitude = ((code & 0x7fu) == 0x7fu) ? 0 : int(m | (e != 0u ? 8u : 0u));
  significand = (code & 0x80u) != 0u ? -magnitude : magnitude;
  exponent = e != 0u ? int(e) - 7 : -6;
}

// A finite half as value = significand * 2^(exponent - 10).
void f16Split(uint bits, out int significand, out int exponent) {
  uint e = (bits >> 10u) & 31u, m = bits & 0x3ffu;
  int magnitude = int(m | (e != 0u ? 0x400u : 0u));
  significand = (bits & 0x8000u) != 0u ? -magnitude : magnitude;
  exponent = e != 0u ? int(e) - 15 : -14;
}

// Code `index` (0..15) of sixteen E4M3 codes packed little-endian in four words.
uint e4m3Code(uint4 codes, uint index) { return (codes[index >> 2u] >> ((index & 3u) * 8u)) & 0xffu; }

// One F13 group: sixteen E4M3 products into an f16 accumulator (half of an m16n8k32 step). Half bits in and out.
uint adaFp8Fdpa16(uint4 a, uint4 b, uint accumulatorBits) {
  if ((accumulatorBits & 0x7c00u) == 0x7c00u) return accumulatorBits;   // an infinite or NaN C passes through
  int maximumExponent = -21;
  int accumulatorSignificand = 0, accumulatorExponent = 0;
  if ((accumulatorBits & 0x7fffu) != 0u) {
    f16Split(accumulatorBits, accumulatorSignificand, accumulatorExponent);
    maximumExponent = accumulatorExponent;
  }
  int products[16], exponents[16];
  uint i;
  for (i = 0u; i < 16u; ++i) {
    int sa, ea, sb, eb;
    e4m3Split(e4m3Code(a, i), sa, ea);
    e4m3Split(e4m3Code(b, i), sb, eb);
    products[i] = sa * sb;
    exponents[i] = ea + eb;
    if (products[i] != 0) maximumExponent = max(maximumExponent, exponents[i]);
  }
  int units = accumulatorSignificand != 0 ? shiftTowardZero(accumulatorSignificand, accumulatorExponent - maximumExponent + 3) : 0;
  for (i = 0u; i < 16u; ++i)
    if (products[i] != 0) units += shiftTowardZero(products[i], exponents[i] - maximumExponent + 7);
  // |units| < 2^24 and the scale is an exact power of two, so the product is the exact sum; one rounding to half.
  return f16BitsExact(roundF16(float(units) * powerOfTwo(maximumExponent - 13)));
}

// f16 + f16 rounded to half (the partition sums): the f32 sum, then RNE to half.
uint addHalf(uint a, uint b) { return f16BitsExact(roundF16(f16ToF32(a) + f16ToF32(b))); }

#endif
