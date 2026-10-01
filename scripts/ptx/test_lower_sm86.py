"""Exhaustive checks of lower_sm86.py on the GPU, through the driver's CUDA library (no CUDA toolkit).

  python scripts/ptx/test_lower_sm86.py

Three tiny kernels are written with the FP8 instructions the generators use, lowered by lower_sm86.py and loaded
with cuModuleLoadData (the driver compiles the PTX for the GPU it runs on):

* decode: cvt.rn.f16x2.e4m3x2 over every pair of codes (65536), against the exact E4M3 values;
* encode: cvt.rn.satfinite.e4m3x2.f16x2 over every half in both lanes, against round-to-nearest-even with
  saturation at 448 and NaN -> 0x7f;
* mma: m16n8k32 E4M3 MMAs on random fragments, against the F13 group arithmetic of docs/numerics.md. The lowered
  MMA is not expected to be bit-exact (Ampere sums each group differently); the check is that it computes the same
  products and groups: every result within the exponent-based local summation bound below. This is not a
  full-network error bound.

Needs NumPy and an NVIDIA GPU of compute capability 8.0 or later.
"""
from __future__ import annotations

import ctypes as ct
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lower_sm86 import lower  # noqa: E402

HEADER = ".version 8.7\n.target sm_89\n.address_size 64\n"

DECODE_PTX = HEADER + """
.visible .entry decode(.param .u64 pOut) {
  .reg .b64 %rd<4>; .reg .b32 %r<4>; .reg .b16 %h<2>;
  ld.param.u64 %rd0, [pOut];
  mov.u32 %r0, %ctaid.x; mov.u32 %r1, %tid.x; mad.lo.u32 %r0, %r0, 256, %r1;
  cvt.u16.u32 %h0, %r0;
  cvt.rn.f16x2.e4m3x2 %r2, %h0;
  mul.wide.u32 %rd1, %r0, 4; add.u64 %rd2, %rd0, %rd1; st.global.b32 [%rd2], %r2;
  ret;
}
"""

ENCODE_PTX = HEADER + """
.visible .entry encode(.param .u64 pIn, .param .u64 pOut) {
  .reg .b64 %rd<6>; .reg .b32 %r<4>; .reg .b16 %h<2>;
  ld.param.u64 %rd0, [pIn]; ld.param.u64 %rd3, [pOut];
  mov.u32 %r0, %ctaid.x; mov.u32 %r1, %tid.x; mad.lo.u32 %r0, %r0, 256, %r1;
  mul.wide.u32 %rd1, %r0, 4; add.u64 %rd2, %rd0, %rd1; ld.global.b32 %r2, [%rd2];
  cvt.rn.satfinite.e4m3x2.f16x2 %h0, %r2;
  mul.wide.u32 %rd1, %r0, 2; add.u64 %rd4, %rd3, %rd1; st.global.b16 [%rd4], %h0;
  ret;
}
"""

# One warp per MMA: A (4 regs), B (2 regs), C (2 regs) per lane, laid out lane-major; D (2 regs) per lane out.
MMA_PTX = HEADER + """
.visible .entry mma(.param .u64 pA, .param .u64 pB, .param .u64 pC, .param .u64 pD) {
  .reg .b64 %rd<12>; .reg .b32 %r<16>;
  ld.param.u64 %rd0, [pA]; ld.param.u64 %rd1, [pB]; ld.param.u64 %rd2, [pC]; ld.param.u64 %rd3, [pD];
  mov.u32 %r0, %ctaid.x; mov.u32 %r1, %tid.x; mad.lo.u32 %r0, %r0, 32, %r1;
  mul.wide.u32 %rd4, %r0, 16; add.u64 %rd5, %rd0, %rd4; ld.global.v4.b32 {%r2, %r3, %r4, %r5}, [%rd5];
  mul.wide.u32 %rd4, %r0, 8; add.u64 %rd6, %rd1, %rd4; ld.global.v2.b32 {%r6, %r7}, [%rd6];
  add.u64 %rd7, %rd2, %rd4; ld.global.v2.b32 {%r8, %r9}, [%rd7];
  mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r10, %r11}, {%r2, %r3, %r4, %r5}, {%r6, %r7}, {%r8, %r9};
  add.u64 %rd8, %rd3, %rd4; st.global.v2.b32 [%rd8], {%r10, %r11};
  ret;
}
"""


class Cuda:
    def __init__(self):
        name = "nvcuda.dll" if os.name == "nt" else "libcuda.so.1"
        self.lib = ct.WinDLL(name) if os.name == "nt" else ct.CDLL(name)
        self.call("cuInit", 0)
        device = ct.c_int()
        self.call("cuDeviceGet", ct.byref(device), 0)
        major, minor = ct.c_int(), ct.c_int()
        self.call("cuDeviceGetAttribute", ct.byref(major), 75, device)
        self.call("cuDeviceGetAttribute", ct.byref(minor), 76, device)
        self.capability = (major.value, minor.value)
        self.context = ct.c_void_p()
        self.call("cuCtxCreate_v2", ct.byref(self.context), 0, device)

    def call(self, function, *args):
        result = getattr(self.lib, function)(*args)
        if result != 0:
            raise RuntimeError(f"{function} failed with CUDA error {result}")

    def function(self, ptx: str, entry: str):
        module, kernel = ct.c_void_p(), ct.c_void_p()
        self.call("cuModuleLoadData", ct.byref(module), ct.c_char_p(ptx.encode() + b"\0"))
        self.call("cuModuleGetFunction", ct.byref(kernel), module, entry.encode())
        return kernel

    def upload(self, array: np.ndarray):
        pointer = ct.c_uint64()
        self.call("cuMemAlloc_v2", ct.byref(pointer), ct.c_size_t(max(array.nbytes, 4)))
        self.call("cuMemcpyHtoD_v2", pointer, array.ctypes.data_as(ct.c_void_p), ct.c_size_t(array.nbytes))
        return pointer

    def empty(self, nbytes: int):
        pointer = ct.c_uint64()
        self.call("cuMemAlloc_v2", ct.byref(pointer), ct.c_size_t(nbytes))
        return pointer

    def download(self, pointer, dtype, count):
        out = np.empty(count, dtype)
        self.call("cuMemcpyDtoH_v2", out.ctypes.data_as(ct.c_void_p), pointer, ct.c_size_t(out.nbytes))
        return out

    def launch(self, kernel, grid: int, block: int, *pointers):
        values = [ct.c_uint64(p.value) for p in pointers]
        params = (ct.c_void_p * len(values))(*[ct.cast(ct.byref(v), ct.c_void_p) for v in values])
        self.call("cuLaunchKernel", kernel, grid, 1, 1, block, 1, 1, 0, None, params, None)
        self.call("cuCtxSynchronize")


def e4m3_values() -> np.ndarray:
    codes = np.arange(256)
    exponent, mantissa = (codes >> 3) & 15, codes & 7
    value = np.where(exponent == 0, mantissa / 512.0, (1 + mantissa / 8.0) * np.exp2(exponent - 7.0))
    return np.where(codes & 0x80, -value, value)


def encode_reference(half_bits: np.ndarray) -> np.ndarray:
    """RNE half -> E4M3 with finite saturation (0x7e) and NaN -> 0x7f (unsigned), for every half pattern."""
    values = half_bits.view(np.float16).astype(np.float64)
    magnitude = np.abs(np.nan_to_num(values, nan=0.0))
    table = e4m3_values()[:127]                      # finite non-negative codes 0x00..0x7e
    index = np.searchsorted(table, magnitude)        # first code >= magnitude
    index = np.clip(index, 1, 126)
    low, high = table[index - 1], table[index]
    pick_high = (magnitude - low > high - magnitude) | ((magnitude - low == high - magnitude) & (index % 2 == 0))
    code = np.where(pick_high, index, index - 1)
    code = np.where(magnitude >= 448.0, 0x7e, code)
    code = np.where(magnitude <= table[0], 0, code)
    code = code | np.where(np.signbit(values), 0x80, 0)
    return np.where(np.isnan(values), 0x7f, code).astype(np.uint16)


def f13_group(a: np.ndarray, b: np.ndarray, accumulator: float) -> float:
    """ref::adaFp8Fdpa16: one group of 16 E4M3 products into an f16 accumulator (F13 fixed point, truncated)."""
    def exponent(v, low):
        return max(int(np.floor(np.log2(abs(v)))) if v != 0 else low, low)
    top = -21 if accumulator == 0 else exponent(accumulator, -14)
    for x, y in zip(a, b):
        if x != 0 and y != 0:
            top = max(top, exponent(x, -6) + exponent(y, -6))
    total = 0.0
    if accumulator != 0:
        e = exponent(accumulator, -14)
        total += np.trunc(accumulator * 2.0 ** -e * 2.0 ** (e - top + 13)) / 8192.0
    for x, y in zip(a, b):
        if x != 0 and y != 0:
            ex, ey = exponent(x, -6), exponent(y, -6)
            total += np.trunc((x * 2.0 ** -ex) * (y * 2.0 ** -ey) * 2.0 ** (ex + ey - top + 13)) / 8192.0
    return float(np.float16(total * 2.0 ** top))


def main() -> int:
    cuda = Cuda()
    print(f"GPU compute capability {cuda.capability[0]}.{cuda.capability[1]}")
    values = e4m3_values()
    failures = 0

    # decode: every pair of codes
    lowered, _ = lower(DECODE_PTX)
    out = cuda.empty(65536 * 4)
    cuda.launch(cuda.function(lowered, "decode"), 256, 256, out)
    decoded = cuda.download(out, np.uint32, 65536)
    halves = np.stack([(decoded & 0xffff).astype(np.uint16), (decoded >> 16).astype(np.uint16)], 1).view(np.float16)
    codes = np.arange(65536)
    expected = np.stack([values[codes & 0xff], values[codes >> 8]], 1)
    finite = ((codes & 0x7f) != 0x7f)[:, None] & ((codes >> 8 & 0x7f) != 0x7f)[:, None]
    same = (halves.astype(np.float64) == expected) & (np.signbit(halves) == np.signbit(expected))
    bad = np.count_nonzero(~same & finite)
    print(f"decode: {65536 - np.count_nonzero(~finite.ravel())} code pairs without the NaN code, {bad} wrong")
    failures += bad

    # encode: every half in the low lane (with the bit-reversed index in the high lane), and the reverse
    lowered, _ = lower(ENCODE_PTX)
    kernel = cuda.function(lowered, "encode")
    low = np.arange(65536, dtype=np.uint32)
    high = ((low * 40503) & 0xffff).astype(np.uint32)
    for name, packed in (("low lane", low | (high << 16)), ("high lane", high | (low << 16))):
        src, dst = cuda.upload(packed.astype(np.uint32)), cuda.empty(65536 * 2)
        cuda.launch(kernel, 256, 256, src, dst)
        got = cuda.download(dst, np.uint16, 65536)
        want = encode_reference((packed & 0xffff).astype(np.uint16)) | (encode_reference((packed >> 16).astype(np.uint16)) << 8)
        bad = np.count_nonzero(got != want)
        print(f"encode ({name}): 65536 halves, {bad} wrong")
        failures += bad

    # mma: random fragments
    rng = np.random.default_rng(1)
    warps = 2000
    def nearest_codes(v):
        """Nearest finite E4M3 code of each value (activation- and weight-like magnitudes, some exact zeros)."""
        table = values[:127]
        m = np.minimum(np.abs(v), 448.0)
        i = np.clip(np.searchsorted(table, m), 1, 126)
        code = np.where(m - table[i - 1] > table[i] - m, i, i - 1)
        return (code | np.where(v < 0, 0x80, 0)).astype(np.uint8)
    spread = lambda shape: rng.standard_normal(shape) * np.exp2(rng.uniform(-6, 2, shape))
    a_codes = nearest_codes(spread((warps, 16, 32)) * (rng.random((warps, 16, 32)) > 0.1))
    b_codes = nearest_codes(spread((warps, 32, 8)) * 0.25)
    c_vals = (rng.standard_normal((warps, 16, 8)) * 4).astype(np.float16)
    # fragment packing (PTX ISA, m16n8k32 .e4m3): A a0..a3 = rows g, g+8, g, g+8 / k 4t..4t+3, +16 for a2, a3
    a_frag = np.zeros((warps, 32, 16), np.uint8)
    b_frag = np.zeros((warps, 32, 8), np.uint8)
    c_frag = np.zeros((warps, 32, 4), np.float16)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for i in range(16):
            row = g + (8 if (i // 4) % 2 else 0)
            k = 4 * t + (i % 4) + (16 if i >= 8 else 0)
            a_frag[:, lane, i] = a_codes[:, row, k]
        for i in range(8):
            k = 4 * t + (i % 4) + (16 if i >= 4 else 0)
            b_frag[:, lane, i] = b_codes[:, k, g]
        for i in range(4):
            row, col = g + (8 if i >= 2 else 0), 2 * t + (i % 2)
            c_frag[:, lane, i] = c_vals[:, row, col]
    lowered, _ = lower(MMA_PTX)
    pa, pb, pc = cuda.upload(a_frag), cuda.upload(b_frag), cuda.upload(c_frag.view(np.uint16))
    pd = cuda.empty(warps * 32 * 8)
    cuda.launch(cuda.function(lowered, "mma"), warps, 32, pa, pb, pc, pd)
    d_frag = cuda.download(pd, np.uint16, warps * 32 * 4).view(np.float16).reshape(warps, 32, 4)
    # A wrong pairing of products would be off by about a product (2^E, E the group's largest product exponent);
    # a different summation of the same products is off by at most the F13 truncations: 17 units of 2^(E - 13) per
    # group, plus the half rounding. The bound below is 2^(E - 7) + one ULP, per two chained groups.
    exact = within = total = 0
    worst = 0.0
    def top_exponent(av, bv, accumulator):
        e = -21 if accumulator == 0 else max(int(np.floor(np.log2(abs(accumulator)))), -14)
        for x, y in zip(av, bv):
            if x != 0 and y != 0:
                e = max(e, max(int(np.floor(np.log2(abs(x)))), -6) + max(int(np.floor(np.log2(abs(y)))), -6))
        return e
    for w in range(0, warps, 10):
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for i in range(4):
                row, col = g + (8 if i >= 2 else 0), 2 * t + (i % 2)
                av, bv = values[a_codes[w, row]], values[b_codes[w, :, col]]
                c = float(c_vals[w, row, col])
                first = f13_group(av[:16], bv[:16], c)
                ref = f13_group(av[16:], bv[16:], first)
                got = float(d_frag[w, lane, i])
                e = max(top_exponent(av[:16], bv[:16], c), top_exponent(av[16:], bv[16:], first))
                bound = 2.0 ** (e - 7) + float(np.spacing(np.float16(abs(ref))))
                total += 1
                exact += got == ref
                within += abs(got - ref) <= bound
                worst = max(worst, abs(got - ref) / 2.0 ** (e - 13))
    print(f"mma: {total} results, {exact} equal to the F13 reference ({100 * exact / total:.1f}%), "
          f"{within} within the summation bound ({100 * within / total:.2f}%), worst {worst:.1f} F13 units")
    if within != total:
        failures += total - within
    print("PASS" if failures == 0 else f"FAIL ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
