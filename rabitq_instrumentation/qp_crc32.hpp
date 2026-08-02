// qp_crc32.hpp — CRC-32/ISO-HDLC kernels for the Option-B detection layer.
//
// WHY THIS FILE EXISTS, AND WHY IT IS NOT IN recovery-changes.patch
// -----------------------------------------------------------------
// The detection layer's cost is dominated by this one function. Measured on meow1 with the
// original byte-at-a-time kernel: on-access detection costs +7.78% of search wall for
// `fallback_eb` and +209.53% for `drop` (docs/claim_impl_map.md, "Detection cost relative to
// NO detection"). That is a property of the KERNEL, not of on-access detection as such, so the
// kernel is worth making swappable and measuring.
//
// recovery-changes.patch is 324 hand-maintained lines with @@ anchors and no `index` line (so
// `git apply --3way` is unavailable). Growing it by ~200 lines of SIMD would be a liability.
// Instead the kernels live here as a normal reviewable header, copied into
// <lib>/include/rabitqlib/ by build_rabitq.sh, and the patch grows by ~10 lines: an #include,
// an `fi_crc_impl_` member, and a dispatcher.
//
// THE NON-NEGOTIABLE PROPERTY
// ---------------------------
// All kernels must be BIT-IDENTICAL to `zlib.crc32` (CRC-32/ISO-HDLC: poly 0xEDB88320
// reflected, init 0xFFFFFFFF, final xor 0xFFFFFFFF). That is what qp/rabitq/crc_manifest.py
// writes into every manifest (ALGO_CRC32 = 1). Bit-identity is what lets a kernel swap leave
// every frozen manifest, every frozen expb record and every gate result valid, with nothing to
// re-run. It is proven exhaustively by artifacts/phase3/tests/test_crc_kernel.py, not asserted
// here. SSE4.2's _mm_crc32_u64 computes CRC-32C (Castagnoli) — a DIFFERENT polynomial — and is
// therefore unusable no matter how fast it is; PCLMULQDQ is the hardware route that preserves
// the polynomial.
//
// `n` is a RUNTIME value. The ex window is 96 B on SIFT b=7, but that comes from the manifest
// header (field_len), not a compile-time constant, and the loader self-test passes 9. Every
// kernel must handle arbitrary n including 0.
//
// ISA guarding follows the surrounding house style: compile-time `#if defined(__FEATURE__)`,
// never runtime dispatch (rabitqlib has zero __builtin_cpu_supports / cpuid call sites outside
// third/). Deployment already requires AVX512BW for RaBitQ's search path, a strict superset of
// PCLMUL, so the guard is belt-and-braces rather than a real portability path.

#ifndef QP_CRC32_HPP_
#define QP_CRC32_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__PCLMUL__)
#include <immintrin.h>
#endif

namespace qp_crc {

enum Impl {
    kTable = 0,   // byte-at-a-time, 1 KB table — the original fi_crc32, kept as the control
    kSlice8 = 1,  // slicing-by-8, 8 KB table — portable, no ISA requirement
    kClmul = 2,   // PCLMULQDQ folding — same polynomial, so still bit-identical
    kImplCount = 3
};

// ---------------------------------------------------------------------------
// Shared tables. Generated once at first use from the reflected polynomial, so there is no
// 8 KB literal blob to mistype and no chance of the table disagreeing with the polynomial it
// claims to implement.
// ---------------------------------------------------------------------------

struct Tables {
    uint32_t t[8][256];

    Tables() {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) {
                c = ((c & 1U) != 0) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
            }
            t[0][i] = c;
        }
        // t[k][i] advances t[k-1][i] by one more zero byte. This is the slicing-by-8
        // construction; t[0] alone is exactly the classic table.
        for (uint32_t i = 0; i < 256; i++) {
            for (int k = 1; k < 8; k++) {
                t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 0xFFU];
            }
        }
    }
};

inline const Tables& tables() {
    static const Tables tbl;
    return tbl;
}

// ---------------------------------------------------------------------------
// kTable — the original kernel, moved verbatim so the control arm is genuinely the old code.
// ---------------------------------------------------------------------------

// Raw update: no init, no final xor, so it composes. The public entry points wrap it.
inline uint32_t update_table(uint32_t crc, const unsigned char* p, size_t n) {
    const Tables& tb = tables();
    for (size_t i = 0; i < n; i++) {
        crc = tb.t[0][(crc ^ p[i]) & 0xFFU] ^ (crc >> 8);
    }
    return crc;
}

inline uint32_t crc32_table(const void* data, size_t n) {
    return update_table(0xFFFFFFFFU, static_cast<const unsigned char*>(data), n) ^ 0xFFFFFFFFU;
}

// ---------------------------------------------------------------------------
// kSlice8 — consumes 8 bytes per iteration through 8 independent table lookups, breaking the
// per-byte dependency chain that limits kTable. x86-64 only (little-endian assumed); the
// project's build_rabitq.sh hard-stops on non-x86_64 anyway. Loads go through memcpy to avoid
// unaligned/strict-aliasing UB — it compiles to a single mov.
// ---------------------------------------------------------------------------

inline uint32_t update_slice8(uint32_t crc, const unsigned char* p, size_t n) {
    const Tables& tb = tables();
    while (n >= 8) {
        uint32_t lo, hi;
        std::memcpy(&lo, p, 4);
        std::memcpy(&hi, p + 4, 4);
        lo ^= crc;
        crc = tb.t[7][lo & 0xFFU] ^ tb.t[6][(lo >> 8) & 0xFFU] ^ tb.t[5][(lo >> 16) & 0xFFU] ^
              tb.t[4][(lo >> 24) & 0xFFU] ^ tb.t[3][hi & 0xFFU] ^ tb.t[2][(hi >> 8) & 0xFFU] ^
              tb.t[1][(hi >> 16) & 0xFFU] ^ tb.t[0][(hi >> 24) & 0xFFU];
        p += 8;
        n -= 8;
    }
    for (size_t i = 0; i < n; i++) {
        crc = tb.t[0][(crc ^ p[i]) & 0xFFU] ^ (crc >> 8);
    }
    return crc;
}

inline uint32_t crc32_slice8(const void* data, size_t n) {
    return update_slice8(0xFFFFFFFFU, static_cast<const unsigned char*>(data), n) ^ 0xFFFFFFFFU;
}

// ---------------------------------------------------------------------------
// kClmul — PCLMULQDQ folding, per Intel white paper 323102 (Gopal et al., 2009) §"Bit-Reflection".
//
// CONSTANTS. Reflected-domain fold constants for one 128-bit step:
//     K_LO = 0x1751997D0 = reflect32(x^(128+32) mod P) << 1
//     K_HI = 0x0CCAA009E = reflect32(x^(128-32) mod P) << 1
// The `<< 1` compensates for the extra factor of x that carry-less multiply introduces in the
// lsb-first domain (a 64x64 clmul puts its 127-bit product in physical bits 0..126, which in
// the reflected reading are the coefficients of x^1..x^127, not x^0..x^126).
//
// These were NOT copied from any implementation. They were obtained by searching the published
// reflected constant set for the (lane, exponent) assignment that makes a pure-Python model of
// this exact fold reproduce zlib.crc32, then confirmed against Intel 323102 p.22. That matters
// because the constant conventions are NOT interchangeable between references -- Linux's
// constant for this same fold distance is 0x00000000AE689191 (x^159, no pre-shift), and
// 0x1751997D0 >> 1 != 0xAE689191, since reflection does not commute with shifting. Constants,
// clmul immediates and the reduction step are one atomic unit; mixing sources silently yields
// wrong CRCs. docs/crc_kernel.md records the derivation.
//
// NO BARRETT REDUCTION. Every reference implementation finishes the finail 128 bits with a
// Barrett step (2-3 serially dependent clmuls). We instead run the folded accumulator's 16
// bytes back through slice8 from crc=0. That is exact -- the accumulator IS a 16-byte message
// whose CRC is the answer once the init constant has been folded in -- and on Zen 5, where
// PCLMULQDQ is 4 uops at one per 2 cycles (uops.info), trading ~3 serial clmuls for 2 slice8
// iterations is the right side of the trade. It also removes the mu/P constants entirely, so
// there is that much less convention to get wrong.
// ---------------------------------------------------------------------------

#if defined(__PCLMUL__)
inline uint32_t crc32_clmul(const void* data, size_t n) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    // Below one full block there is nothing to fold; slice8 is also what the 9-byte loader
    // self-test hits, so that path is exercised on every load.
    if (n < 16) return update_slice8(0xFFFFFFFFU, p, n) ^ 0xFFFFFFFFU;

    __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    x = _mm_xor_si128(x, _mm_cvtsi32_si128(static_cast<int>(0xFFFFFFFFU)));  // fold in the init
    p += 16;
    n -= 16;

    // Lane 0 (low) = K_LO, lane 1 (high) = K_HI: the earlier bytes sit in the low qword and so
    // must advance by the LARGER power of x.
    const __m128i k = _mm_set_epi64x(static_cast<int64_t>(0x0CCAA009EULL),
                                     static_cast<int64_t>(0x1751997D0ULL));
    while (n >= 16) {
        const __m128i nxt = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        const __m128i t0 = _mm_clmulepi64_si128(x, k, 0x00);  // x.lo * K_LO
        const __m128i t1 = _mm_clmulepi64_si128(x, k, 0x11);  // x.hi * K_HI
        x = _mm_xor_si128(_mm_xor_si128(t0, t1), nxt);
        p += 16;
        n -= 16;
    }

    unsigned char acc[16];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(acc), x);
    uint32_t crc = update_slice8(0U, acc, 16);
    crc = update_slice8(crc, p, n);
    return crc ^ 0xFFFFFFFFU;
}
#else
inline uint32_t crc32_clmul(const void* data, size_t n) { return crc32_table(data, n); }
#endif

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

inline bool impl_available(int impl) {
    if (impl == kClmul) {
#if defined(__PCLMUL__)
        return true;
#else
        return false;
#endif
    }
    return impl >= 0 && impl < kImplCount;
}

inline const char* impl_name(int impl) {
    switch (impl) {
        case kTable: return "table";
        case kSlice8: return "slice8";
        case kClmul: return "clmul";
        default: return "unknown";
    }
}

// Returns -1 on an unknown name so callers report-and-stop rather than silently defaulting —
// a mistyped --crc-impl must not quietly measure the wrong kernel.
inline int impl_from_name(const char* s) {
    if (s == nullptr) return -1;
    if (std::strcmp(s, "table") == 0) return kTable;
    if (std::strcmp(s, "slice8") == 0) return kSlice8;
    if (std::strcmp(s, "clmul") == 0) return kClmul;
    return -1;
}

inline uint32_t crc32(int impl, const void* data, size_t n) {
    switch (impl) {
        case kSlice8: return crc32_slice8(data, n);
        case kClmul: return crc32_clmul(data, n);
        default: return crc32_table(data, n);
    }
}

// The loader's self-test vector, kept next to the kernels so every impl is checked with it.
inline bool self_test(int impl) { return crc32(impl, "123456789", 9) == 0xCBF43926U; }

}  // namespace qp_crc

#endif  // QP_CRC32_HPP_
