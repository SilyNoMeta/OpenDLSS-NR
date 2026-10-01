"""Lower the generated E4M3 PTX kernels to Ampere (sm_80 and up, validated on sm_86): no FP8 instruction is left.

  python scripts/ptx/lower_sm86.py <input .ptx | directory> <output .ptx | directory> [--accumulate=f16|f32]

The kernels of scripts/ptx use exactly three FP8 instructions, all sm_89+. Each becomes a scoped block of
sm_80-compatible instructions:

* mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 -> two mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
  on decoded operands. An E4M3 value is always exactly an f16 value, so the decode is exact. The k32 step's two
  k16 halves stay the two f16 MMAs' K dimensions in order (k 0..15, then 16..31). Within a half, the four E4M3
  codes a thread holds (k = 4t..4t+3) occupy the f16 fragment's k slots 2t, 2t+1, 2t+8, 2t+9, for A and for B
  alike, so every product pairs the same two elements and every 16-product group sums the same 16 products.
  The C/D fragments of the two shapes are laid out identically.

  What is NOT preserved is the addition: Ada sums each group of 16 products as an F13 fixed-point dot product
  (docs/numerics.md); Ampere's f16 MMA sums them its own way. The lowered kernels are therefore a close
  approximation of the native ones, not a bit-exact copy (docs/ampere: on random E4M3 data 74.6% of groups give
  the same half, 95.4% within one ULP). The exact route on Ampere is the compatibility backend.

* cvt.rn.f16x2.e4m3x2 (decode two codes) -> the bit-level decode below; exact for every code but the NaN code
  0x7f / 0xff (decoded as 480 instead of NaN), which no activation or weight of the network holds (activations
  publish a NaN as +0, and the model loader replaces NaN weight codes).

* cvt.rn.satfinite.e4m3x2.f16x2 (encode two halves) -> integer round-to-nearest-even with finite saturation at
  448 (code 0x7e) and NaN -> 0x7f, the instruction's own semantics, for every half input.

`.target sm_89` becomes `.target sm_80`: nothing left needs more than Ampere's first generation (f16 mma.sync,
cp.async, ldmatrix), so the driver can compile the kernels for any compute capability 8.x; they are validated on
sm_86 only. Nothing else changes: register caps, shared memory and launch shapes are
the Ada generators' choices, which an Ampere-tuned generator may later revisit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MMA = re.compile(
    r"^(?P<indent>\s*)mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+"
    r"\{(?P<d>[^}]*)\},\s*\{(?P<a>[^}]*)\},\s*\{(?P<b>[^}]*)\},\s*\{(?P<c>[^}]*)\};\s*$")
DECODE = re.compile(r"^(?P<indent>\s*)cvt\.rn\.f16x2\.e4m3x2\s+(?P<d>%\w+),\s*(?P<a>%\w+);\s*$")
ENCODE = re.compile(r"^(?P<indent>\s*)cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+(?P<d>%\w+),\s*(?P<a>%\w+);\s*$")


def regs(text: str) -> list[str]:
    return [r.strip() for r in text.split(",")]


def decode_pair(source: str, selector: str, out: str, t: str, u: str, scale: str) -> list[str]:
    """Two E4M3 codes (bytes picked by `selector` from `source`) -> f16x2 in `out`.

    Each code is spread to the low byte of a 16-bit lane; (code & 0x7f) << 7 is then the f16 bit pattern of
    value * 2^-8 for every code (normal codes land on normal halves, subnormal codes on subnormal halves), the
    sign moves to bit 15, and one exact f16 multiply by 256 restores the scale.
    """
    return [
        f"prmt.b32 {t}, {source}, 0, {selector};",
        f"shl.b32 {u}, {t}, 7;",
        f"and.b32 {u}, {u}, 0x3f803f80;",
        f"shl.b32 {t}, {t}, 8;",
        f"and.b32 {t}, {t}, 0x80008000;",
        f"or.b32 {u}, {u}, {t};",
        f"mul.rn.f16x2 {out}, {u}, {scale};",
    ]


ACCUMULATE = "f16"   # set by main(): "f16" (one f16 MMA per group) or "f32" (f32 MMA, then RN to half per group)


def unpack_f16x2_to_f32(source: str, low: str, high: str) -> list[str]:
    return [f"mov.b32 {{%lo_hl, %lo_hh}}, {source};", f"cvt.f32.f16 {low}, %lo_hl;", f"cvt.f32.f16 {high}, %lo_hh;"]


def pack_f32_to_f16x2(low: str, high: str, out: str) -> list[str]:
    # cvt.rn.f16x2.f32 d, a, b packs a into the upper half and b into the lower one
    return [f"cvt.rn.f16x2.f32 {out}, {high}, {low};"]


def lower_mma(m: re.Match) -> list[str]:
    d, a, b, c = regs(m["d"]), regs(m["a"]), regs(m["b"]), regs(m["c"])
    assert len(d) == 2 and len(a) == 4 and len(b) == 2 and len(c) == 2, m.group(0)
    body = [".reg .b32 %lo_a<8>, %lo_b<4>, %lo_t, %lo_u, %lo_s, %lo_m<2>;", "mov.b32 %lo_s, 0x5c005c00;"]
    if ACCUMULATE == "f32":
        body.append(".reg .f32 %lo_f<4>; .reg .b16 %lo_hl, %lo_hh;")
    # A: rows g (a0, a2) and g+8 (a1, a3); bytes 0,1 -> f16 k slots 2t, 2t+1; bytes 2,3 -> slots 2t+8, 2t+9
    for half, (rg, rg8) in enumerate(((a[0], a[1]), (a[2], a[3]))):
        o = 4 * half
        body += decode_pair(rg, "0x4140", f"%lo_a{o + 0}", "%lo_t", "%lo_u", "%lo_s")
        body += decode_pair(rg8, "0x4140", f"%lo_a{o + 1}", "%lo_t", "%lo_u", "%lo_s")
        body += decode_pair(rg, "0x4342", f"%lo_a{o + 2}", "%lo_t", "%lo_u", "%lo_s")
        body += decode_pair(rg8, "0x4342", f"%lo_a{o + 3}", "%lo_t", "%lo_u", "%lo_s")
    for half, rb in enumerate(b):
        body += decode_pair(rb, "0x4140", f"%lo_b{2 * half + 0}", "%lo_t", "%lo_u", "%lo_s")
        body += decode_pair(rb, "0x4342", f"%lo_b{2 * half + 1}", "%lo_t", "%lo_u", "%lo_s")
    if ACCUMULATE == "f16":
        body.append("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%lo_m0, %lo_m1}, "
                    "{%lo_a0, %lo_a1, %lo_a2, %lo_a3}, {%lo_b0, %lo_b1}, " + "{" + f"{c[0]}, {c[1]}" + "};")
        body.append("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {" + f"{d[0]}, {d[1]}" + "}, "
                    "{%lo_a4, %lo_a5, %lo_a6, %lo_a7}, {%lo_b2, %lo_b3}, {%lo_m0, %lo_m1};")
        return body
    # f32 accumulation, published to half after each group of 16 products as the native F13 step publishes its
    # accumulator: the C/D fragment element i of an f16 MMA is element i of the f32 one (c0 = elements 0, 1).
    group = lambda a0, b0, source, target: (
        unpack_f16x2_to_f32(source[0], "%lo_f0", "%lo_f1") + unpack_f16x2_to_f32(source[1], "%lo_f2", "%lo_f3") +
        ["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%lo_f0, %lo_f1, %lo_f2, %lo_f3}, "
         f"{{%lo_a{a0}, %lo_a{a0 + 1}, %lo_a{a0 + 2}, %lo_a{a0 + 3}}}, {{%lo_b{b0}, %lo_b{b0 + 1}}}, "
         "{%lo_f0, %lo_f1, %lo_f2, %lo_f3};"] +
        pack_f32_to_f16x2("%lo_f0", "%lo_f1", target[0]) + pack_f32_to_f16x2("%lo_f2", "%lo_f3", target[1]))
    body += group(0, 0, c, ["%lo_m0", "%lo_m1"])
    body += group(4, 2, ["%lo_m0", "%lo_m1"], d)
    return body


def lower_decode(m: re.Match) -> list[str]:
    body = [".reg .b32 %lo_t, %lo_u, %lo_s, %lo_x;", "mov.b32 %lo_s, 0x5c005c00;",
            f"cvt.u32.u16 %lo_x, {m['a']};"]
    body += decode_pair("%lo_x", "0x4140", m["d"], "%lo_t", "%lo_u", "%lo_s")
    return body


def encode_half(source: str, shift: int, out: str) -> list[str]:
    """One half (bits `shift`..`shift`+15 of `source`) -> its E4M3 code in `out` (u32): RNE, satfinite, NaN -> 0x7f.

    Normal E4M3 (half exponent >= 9): the half's magnitude rounded to 3 mantissa bits in one integer step
    ((m + 0x3f + bit7) >> 7, the carry runs into the exponent), minus the bias difference (8 << 3).
    Subnormal E4M3 (half exponent <= 8): the significand shifted right by 16 - e (15 for a subnormal half), RNE.
    """
    return [
        f"bfe.u32 %lo_h, {source}, {shift}, 16;",
        "shr.b32 %lo_g, %lo_h, 8;",
        "and.b32 %lo_g, %lo_g, 0x80;",        # the sign, kept for a zero (0x80 is -0) as the instruction does
        "and.b32 %lo_m, %lo_h, 0x7fff;",
        "shr.b32 %lo_e, %lo_m, 10;",
        # normal path
        "shr.b32 %lo_n, %lo_m, 7;",
        "and.b32 %lo_n, %lo_n, 1;",
        "add.u32 %lo_n, %lo_n, %lo_m;",
        "add.u32 %lo_n, %lo_n, 0x3f;",
        "shr.b32 %lo_n, %lo_n, 7;",
        "sub.u32 %lo_n, %lo_n, 64;",
        "min.u32 %lo_n, %lo_n, 0x7e;",
        # subnormal path: significand and shift
        "and.b32 %lo_q, %lo_m, 0x3ff;",
        "setp.ne.u32 %lo_p, %lo_e, 0;",
        "@%lo_p or.b32 %lo_q, %lo_q, 0x400;",
        "max.u32 %lo_k, %lo_e, 1;",
        "sub.u32 %lo_k, 16, %lo_k;",
        "shr.b32 %lo_r, %lo_q, %lo_k;",
        "shl.b32 %lo_w, 1, %lo_k;",
        "sub.u32 %lo_w, %lo_w, 1;",
        "and.b32 %lo_w, %lo_q, %lo_w;",
        "sub.u32 %lo_k, %lo_k, 1;",
        "shl.b32 %lo_k, 1, %lo_k;",           # halfway
        "and.b32 %lo_z, %lo_r, 1;",
        "add.u32 %lo_z, %lo_z, %lo_w;",       # remainder + odd > halfway  <=>  round up (RNE)
        "setp.gt.u32 %lo_p, %lo_z, %lo_k;",
        "@%lo_p add.u32 %lo_r, %lo_r, 1;",
        "setp.gt.u32 %lo_p, %lo_e, 8;",
        "selp.b32 %lo_n, %lo_n, %lo_r, %lo_p;",
        "setp.gt.u32 %lo_p, %lo_m, 0x7c00;",  # NaN: the canonical NaN code, unsigned
        "selp.b32 %lo_n, 0x7f, %lo_n, %lo_p;",
        "selp.b32 %lo_g, 0, %lo_g, %lo_p;",
        f"or.b32 {out}, %lo_n, %lo_g;",
    ]


def lower_encode(m: re.Match) -> list[str]:
    body = [".reg .b32 %lo_h, %lo_g, %lo_m, %lo_e, %lo_n, %lo_q, %lo_k, %lo_r, %lo_w, %lo_z, %lo_c0, %lo_c1;",
            ".reg .pred %lo_p;"]
    body += encode_half(m["a"], 0, "%lo_c0")
    body += encode_half(m["a"], 16, "%lo_c1")
    body += ["shl.b32 %lo_c1, %lo_c1, 8;", "or.b32 %lo_c0, %lo_c0, %lo_c1;", f"cvt.u16.u32 {m['d']}, %lo_c0;"]
    return body


def lower(text: str) -> tuple[str, dict[str, int]]:
    counts = {"mma": 0, "decode": 0, "encode": 0}
    out = []
    for line in text.splitlines():
        for kind, pattern, rewrite in (("mma", MMA, lower_mma), ("decode", DECODE, lower_decode),
                                       ("encode", ENCODE, lower_encode)):
            m = pattern.match(line)
            if m:
                counts[kind] += 1
                indent = m["indent"]
                out.append(f"{indent}{{  // sm86: {line.strip()}")
                out += [f"{indent}  {instruction}" for instruction in rewrite(m)]
                out.append(f"{indent}}}")
                break
        else:
            if re.match(r"^\s*\.target\s+sm_89\s*$", line):
                line = line.replace("sm_89", "sm_80")
            out.append(line)
    lowered = "\n".join(out) + "\n"
    leftover = re.findall(r"\be4m3\w*|\.target\s+sm_89", re.sub(r"//.*", "", lowered))
    leftover = [w for w in leftover if not w.startswith("e4m3_")]   # entry names such as block32_e4m3_f2
    if leftover:
        raise SystemExit(f"FP8 instruction left after lowering: {sorted(set(leftover))}")
    return lowered, counts


def main() -> None:
    global ACCUMULATE
    arguments = [a for a in sys.argv[1:] if not a.startswith("--accumulate=")]
    for a in sys.argv[1:]:
        if a.startswith("--accumulate="):
            ACCUMULATE = a.split("=", 1)[1]
    if ACCUMULATE not in ("f16", "f32"):
        raise SystemExit("--accumulate=f16 or --accumulate=f32")
    source, target = Path(arguments[0]), Path(arguments[1])
    files = sorted(source.glob("*.ptx")) if source.is_dir() else [source]
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    totals = {"mma": 0, "decode": 0, "encode": 0}
    for path in files:
        lowered, counts = lower(path.read_text(encoding="utf-8"))
        (target / path.name if source.is_dir() else target).write_text(lowered, encoding="utf-8", newline="\n")
        for key in totals:
            totals[key] += counts[key]
    print(f"lowered {len(files)} kernel(s) for sm_80+ ({ACCUMULATE} accumulation): {totals['mma']} E4M3 MMAs, {totals['decode']} decodes, "
          f"{totals['encode']} encodes")


if __name__ == "__main__":
    main()
