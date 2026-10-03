// shaders/preprocess.comp for the Direct3D adapters: input feature generation, sixteen f32 lanes per padded pixel.
// The three Gaussian lanes use the GPU's approximate f32 transcendentals in the GLSL's instruction order; whether
// the Direct3D compiler evaluates them to the same bits as the Vulkan one is measured (tests/d3d), not assumed.
#include "common.hlsl"

cbuffer Push : register(b0) {
  uint fullWidth;
  uint fullHeight;
  uint validWidth;
  uint validHeight;
  uint sourceWidth;
  uint sourceHeight;
  uint seed;
  float autoMask;        // > 0 enables the auto-mask lane pair
  float localTone;
  float localStructure;
  float skinStructure;
  float style;
};

RWByteAddressBuffer proxy : register(u0);      // RGBA f32 proxy code values
RWByteAddressBuffer features : register(u7);   // f32 [full][16]

float hashUniform(uint value) {
  uint mixed = value;
  mixed = (mixed >> ((mixed >> 28u) + 4u)) ^ mixed;
  mixed *= 0x108ef2d9u;
  uint integer = ((mixed >> 30u) ^ (mixed >> 8u)) + 1u;
  return float(integer) * asfloat(0x33800000u);
}

// Scalar form of the network's Box-Muller sequence (instruction order preserved).
float3 gaussian3(uint x, uint y, uint noiseSeed) {
  uint base = (x * 0x8da6b343u) ^ (noiseSeed * 0x9e3779b9u) ^ (y * 0xd8163841u) ^ 0x243f6a88u;
  base = (base >> ((base >> 28u) + 4u)) ^ base;
  base *= 0x108ef2d9u;
  base = (base >> 22u) ^ base;
  float u0 = hashUniform(base * 0x2c9277b5u + 0xac564b05u);
  float u1 = hashUniform(base * 0xfa6dc5f9u + 0x4712a88eu);
  float u2 = hashUniform(base * 0xcaa5b80du + 0x21dd796bu);
  float u3 = hashUniform(base * 0x83232c31u + 0x3463e0acu);
  precise float radius0 = sqrt(log2(u0) * asfloat(0x3f317218u) * -2.0);
  precise float radius1 = sqrt(log2(u2) * asfloat(0x3f317218u) * -2.0);
  precise float angle0 = u1 * asfloat(0x40c90fdbu);
  precise float angle1 = u3 * asfloat(0x40c90fdbu);
  precise float n0 = radius0 * cos(angle0);
  precise float n1 = radius0 * sin(angle0);
  precise float n2 = radius1 * cos(angle1);
  return float3(roundF16(n0), roundF16(n1), roundF16(n2));
}

float centered(float value) {
  // Native: texture sample -> cvt.rn.f16, sub.f16(0.5), mul.f16(0.125).
  precise float shifted = roundF16(value) - 0.5;
  precise float scaled = roundF16(shifted) * 0.125;
  return roundF16(scaled);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  [branch] if (id.x >= fullWidth || id.y >= fullHeight) return;
  // Coordinates outside the valid image are mirrored, while the noise channels still hash the original
  // padded coordinate.
  const uint sourceX = id.x < validWidth ? id.x : 2u * validWidth - id.x - 2u;
  const uint sourceY = id.y < validHeight ? id.y : 2u * validHeight - id.y - 2u;
  const uint imageX = ((2u * sourceX + 1u) * sourceWidth) / (2u * validWidth);
  const uint imageY = ((2u * sourceY + 1u) * sourceHeight) / (2u * validHeight);
  // Beyond one mirror the source coordinate wraps (as in the GLSL) and the texel index leaves the proxy. The Vulkan
  // route reads zeros there and the references record that; a Direct3D 12 root descriptor has no bounds, so the read
  // is refused here instead of faulting the GPU.
  // The load itself is given an address inside the proxy: a conditional expression is evaluated on both sides.
  const uint texel = imageY * sourceWidth + imageX;
  const bool inside = texel < sourceWidth * sourceHeight;
  const uint3 loaded = proxy.Load3((inside ? texel : 0u) * 16u);
  const uint3 rgb = inside ? loaded : uint3(0u, 0u, 0u);
  const float r = centered(asfloat(rgb.x)), g = centered(asfloat(rgb.y)), b = centered(asfloat(rgb.z));
  const float3 noise = gaussian3(id.x, id.y, seed);
  const uint base = (id.y * fullWidth + id.x) * 64u;
  const float structure = roundF16(autoMask > 0.0 ? 1.0 : localStructure);
  const float skin = roundF16(autoMask > 0.0 ? (skinStructure < 0.0 ? localStructure : skinStructure) : -1.0);
  const float masked = roundF16(autoMask > 0.0 ? localStructure : -1.0);
  features.Store4(base, uint4(asuint(noise.x), asuint(noise.y), asuint(noise.z), asuint(1.0)));
  features.Store4(base + 16u, uint4(asuint(r), asuint(g), asuint(b), asuint(r)));
  features.Store4(base + 32u, uint4(asuint(g), asuint(b), asuint(style / 128.0), asuint(roundF16(localTone))));
  features.Store4(base + 48u, uint4(asuint(structure), asuint(skin), asuint(masked), asuint(0.0)));
}
