# The CRC kernel: algorithm, constants, licensing, and what it actually costs

Reference for `rabitq_instrumentation/qp_crc32.hpp`. Written because the detection layer's whole
query-path cost is one function, and because the folding-constant conventions in this area are a
well-known source of silently-wrong results.

## Why this exists

The Option-B detection layer computes CRC-32/ISO-HDLC over a 96 B `ex` window. With the original
byte-at-a-time kernel, on-access detection measured **+7.78%** of search wall for `fallback_eb`
and **+209.53%** for `drop` (`claim_impl_map.md`, "Detection cost relative to NO detection").
Those figures characterise the *kernel*, not on-access detection as such — which is what makes a
faster kernel worth measuring rather than assuming.

## The non-negotiable constraint: bit-identity

`qp/rabitq/crc_manifest.py` writes `zlib.crc32` values (`ALGO_CRC32 = 1`): CRC-32/ISO-HDLC, poly
`0xEDB88320` reflected, init and final-xor `0xFFFFFFFF`. Any kernel must reproduce it **exactly**.
Bit-identity is what makes a kernel swap a no-op for the science: every frozen manifest, expb
record and gate result stays valid, and nothing needs re-running.

This rules out the obvious hardware route. SSE4.2's `_mm_crc32_u64` is hardwired to the
**Castagnoli** polynomial `0x11EDC6F41` (Intel SDM, `CRC32`) — CRC-32C, a different and unrelated
checksum. No combination of init/xorout/reflection converts one into the other. **PCLMULQDQ is
the only x86 hardware route that preserves the polynomial**, which is exactly why Intel's paper is
titled "for *Generic* Polynomials". (On aarch64 the situation reverses: ARMv8's `crc32b/h/w/x`
implement `0x04C11DB7` directly.)

## The algorithm

Intel white paper **323102**, "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ
Instruction", Gopal et al., December 2009.
[Intel (403s to bots)](https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/fast-crc-computation-generic-polynomials-pclmulqdq-paper.pdf)
· [mirror](https://kib.kiev.ua/x86docs/Intel/WhitePapers/323102-XXX.pdf)

The folding identity: split the top 128 bits into halves `H`, `L`, with `G` the remaining `T` bits,

```
M(x) mod P  ≡  { H(x) ⊗ [x^(T+64) mod P] } ⊕ { L(x) ⊗ [x^T mod P] } ⊕ G(x)   (mod P)
```

Each 128-bit fold is 2 `PCLMULQDQ` + 2 `PXOR`. The paper then reduces the final 128 bits with a
Barrett step. **We do not** — see "No Barrett" below.

> **Correction to a widespread claim:** this paper contains **no performance numbers at all** —
> no cycles/byte, no speedup ratio, no benchmark section. Its only claims are qualitative. Figures
> like "1.x cycles/byte" attributed to it almost certainly come from the sibling paper **323405**,
> which is about the SSE4.2 `crc32` instruction and CRC-32**C**. The only buffer-size guidance
> 323102 gives is structural: fold-by-4 needs ≥128 B, fold-by-1 needs ≥32 B.

## The reflected-domain subtlety (where wrong results come from)

CRC-32/ISO-HDLC is LSB-first. PCLMULQDQ itself is bit-order agnostic, so no data reflection is
needed — but a 64×64 carry-less multiply puts its 127-bit product in physical bits 0..126, which
in the lsb-first reading are the coefficients of x¹..x¹²⁷, **not** x⁰..x¹²⁶. Every clmul therefore
implicitly multiplies by an extra factor of x.

Three equivalent compensations exist, and implementations differ in which they pick:

| fix | cost | who |
|---|---|---|
| shift the 128-bit product left by 1 after each clmul | an extra instruction per fold | nobody sensible |
| pre-shift the constant: store `reflect32(x^E mod P) << 1` (a 33-bit constant) | free | Intel, Chromium, zlib-ng |
| use `x^(E-1)` instead of `x^E`, keeping 32-bit constants | free | Linux kernel |

**These conventions are not interchangeable.** For the same 128-bit fold distance, Intel/Chromium
use `0x1751997D0` while Linux uses `0xAE689191` — and `0x1751997D0 >> 1 = 0xBA8CCBE8 ≠ 0xAE689191`,
because reflection does not commute with shifting. Constants, clmul immediates, and the reduction
step are **one atomic unit**; mixing sources produces wrong CRCs that still look plausible.

## Our constants, and how they were established

```
K_LO = 0x1751997D0 = reflect32(x^(128+32) mod P) << 1     lane 0 (low qword)
K_HI = 0x0CCAA009E = reflect32(x^(128-32) mod P) << 1     lane 1 (high qword)
```

Lane assignment: the *earlier* bytes sit in the low qword, so the low qword must advance by the
**larger** power of x. Getting this backwards is the single easiest mistake to make and it fails
only for inputs ≥32 B — a 16 B test passes either way.

**How these were obtained.** Not copied. A pure-Python model of this exact fold was written, and
the (lane, exponent) assignment was solved for by requiring the model to reproduce `zlib.crc32`;
the result was then confirmed against Intel 323102 p.22. The model was validated exhaustively for
every length 0–256 across zeros / ones / random before any C++ was written. That ordering matters:
the constants are *derived from the requirement*, so there is no transcription step to get wrong.

`artifacts/phase3/tests/test_crc_kernel.py` re-proves this against `zlib.crc32` on every build.

## No Barrett reduction

Every reference implementation finishes the last 128 bits with a Barrett step: 2–3 serially
dependent clmuls plus the `µ` and `P'` constants. We instead run the folded accumulator's 16 bytes
back through `slice8` starting from `crc = 0`. This is exact — once the init constant has been
folded into the first block, the accumulator *is* a 16-byte message whose CRC is the answer.

Two reasons it is the better trade here:

1. **Zen 5's PCLMULQDQ is expensive**: 4 µops, reciprocal throughput **2.00 cycles**, latency 4–5
   ([uops.info](https://uops.info/html-instr/PCLMULQDQ_XMM_XMM_I8.html)). Current Intel cores do
   2/cycle at 1 µop — roughly **4× better**. Trading ~3 serial clmuls (~15 cycles) for 2 `slice8`
   iterations (~4 cycles) is clearly right on this host.
2. It removes the `µ` / `P'` constants entirely — that much less convention left to get wrong.

This is also why our result diverges from the literature; see below.

## Licensing: nothing was copied

| implementation | license | usable here? |
|---|---|---|
| [Chromium `crc32_simd.c`](https://github.com/chromium/chromium/blob/main/third_party/zlib/crc32_simd.c) | BSD-3-clause | yes — ~143 self-contained lines, the cleanest to adapt |
| [zlib-ng `crc32_pclmulqdq_tpl.h`](https://github.com/zlib-ng/zlib-ng/blob/develop/arch/x86/crc32_pclmulqdq_tpl.h) | zlib | yes, but 737 lines of template with a folding-state API |
| [Linux `crc-pclmul-template.S`](https://github.com/torvalds/linux/blob/master/lib/crc/x86/crc-pclmul-template.S) | **GPL-2.0** | **no** — useful as documentation only |
| [corsix/fast-crc32](https://github.com/corsix/fast-crc32) | MIT or zlib | yes; a generator, tuned for streaming, no small-input data |

**`qp_crc32.hpp` is original**: derived from the polynomial and validated against `zlib.crc32`, as
described above. The above are cited for the algorithm and for cross-checking the constants, not
used as source. So there is no third-party license obligation in this tree.

## The small-buffer question — and why our answer differs from the literature

Our window is **96 B**, which sits right on the documented crossover:

| implementation | threshold below which folding is not used |
|---|---|
| Chromium zlib (SSE4.2+PCLMUL) | **64 B** (`Z_CRC32_SSE42_MINIMUM_LENGTH`) |
| Chromium zlib (AVX512+VPCLMUL) | 256 B |
| Cloudflare zlib | **~80 B** (measured) |
| Linux `crc32_le` | 16 B |
| Linux `crc32c` | 512 B (but that is vs the `crc32q` *instruction*, and includes kernel FPU save/restore) |

[Fastmail](https://www.fastmail.com/blog/the-search-for-a-faster-crc32/), the one team that
published small-buffer measurements, found *"CloudFlare is amazing until the input buffer gets
under 80 bytes"* and shipped slicing-by-8/16 instead.

On that evidence the expectation for 96 B on Zen 5 was **approximately a wash**. Measurement says
otherwise:

### Measured on meow1 (Ryzen 9 9950X, `-Ofast -march=native`), hot buffer — kernel compute only

| size | `table` | `slice8` | `clmul` | clmul vs table | clmul vs slice8 |
|---|---|---|---|---|---|
| 9 B | 4.4 ns | 1.3 ns | 1.2 ns | 3.7× | 1.1× (falls back to `slice8` below 16 B) |
| **96 B** | **119.8 ns** (1.248 ns/B) | **16.0 ns** (0.167 ns/B) | **5.0 ns** (0.052 ns/B) | **24×** | **3.2×** |
| 1 KB | 1414.8 ns | 242.4 ns | 69.9 ns | 20× | 3.5× |
| 96 MiB | 141.3 ms | 24.7 ms | 8.5 ms | 17× | 2.9× |

### Scattered — 96 B windows at 272 B stride, shuffled, over 280 MB (mirrors `drop`'s pattern)

| | `table` | `slice8` | `clmul` |
|---|---|---|---|
| per window | 221.8 ns | 128.5 ns | **44.9 ns** |

**Why we beat the published crossover.** The thresholds above are properties of *those*
implementations' structure, not of PCLMULQDQ. Chromium's routine collapses four accumulators and
then runs a Barrett reduction — on the order of 13–15 clmuls with a serial chain of 8–10, so its
fixed cost is ~40–60 cycles regardless of input size, which is most of the budget at 96 B. Our
kernel does 5 folds (10 clmuls, chain depth 5) and no Barrett. At Zen 5's 4–5 cycle latency that
predicts ≈25 cycles ≈ 4.4 ns at 5.7 GHz; measured **5.0 ns**. The model and the measurement agree,
which is the reason to believe the number.

**The scattered result also refutes a projection made in `claim_impl_map.md`** — that memory
traffic, not CRC compute, would be the floor for `drop`. If memory dominated, the three kernels
would converge in the scattered test. They do not: `clmul` is still 2.9× faster than `slice8`
there. At 96 B random access, **compute still dominates**. The end-to-end measurement is what
settles what that means for `drop`'s +209.53%.

## Caveats

- One host, one microarchitecture. Zen 5's PCLMULQDQ is unusually *slow* relative to current
  Intel, so this result should, if anything, transfer favourably — but that is untested.
- VPCLMULQDQ (512-bit) is available on this CPU but sustains only 0.5 instructions/cycle at any
  width on Zen 5 ([Mysticial's teardown](https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown/)),
  so it is irrelevant at 96 B — one ZMM op then immediately latency-bound in the tail.
- The hot-buffer column is deliberately cache-resident; it isolates the kernel. Only the
  end-to-end run tells you how much survives into the query path.
