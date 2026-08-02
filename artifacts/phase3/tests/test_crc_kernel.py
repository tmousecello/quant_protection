"""Bit-identity of every qp_crc32.hpp kernel against zlib.crc32.

WHY THIS IS THE LOAD-BEARING TEST
---------------------------------
The detection layer's manifests store `zlib.crc32` values (qp/rabitq/crc_manifest.py,
ALGO_CRC32 = 1). A kernel that is fast but not bit-identical does not "mostly work" -- it
reclassifies elements, which silently changes which candidates `drop` discards and which ones
`fallback_eb` degrades, i.e. it changes recall while every gate still reports green. Worse, it
would invalidate every frozen manifest and expb record. So bit-identity is not a quality bar
here, it is the thing that makes a kernel swap a no-op for the science.

Everything else in this repo checks CRCs at the COUNT level (`elements_crc_fail` vs
`elements_crc_fail_py`). This is the only VALUE-level check, and the only one that would catch
a fold constant that is right for 96 B but wrong for, say, 112 B.

The test compiles the header standalone with g++ -- no RaBitQ tree, no 280 MB index, no AVX512
-- so bit-identity stays testable on any machine rather than only on the workstation.
"""

import os
import shutil
import subprocess
import zlib

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(REPO, "rabitq_instrumentation", "qp_crc32_test.cpp")
HDR = os.path.join(REPO, "rabitq_instrumentation", "qp_crc32.hpp")

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None or not os.path.isfile(SRC),
    reason="g++ or qp_crc32_test.cpp not available",
)


@pytest.fixture(scope="module")
def kernel_bin(tmp_path_factory):
    """Compile the dumper once per session with the project's own optimisation flags."""
    out = tmp_path_factory.mktemp("crc") / "qp_crc32_test"
    proc = subprocess.run(
        ["g++", "-O2", "-march=native", "-std=c++17", "-Wall", "-Wextra", "-o", str(out), SRC],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"compile failed:\n{proc.stdout}\n{proc.stderr}"
    return str(out)


def _run_cases(kernel_bin, tmp_path, data, cases):
    """Return {(impl, off, ln): crc} for every impl the binary was built with."""
    dpath = tmp_path / "data.bin"
    cpath = tmp_path / "cases.txt"
    dpath.write_bytes(data)
    cpath.write_text("".join(f"{o} {n}\n" for o, n in cases))
    proc = subprocess.run([kernel_bin, "--dump", str(dpath), str(cpath)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"dump failed:\n{proc.stdout}\n{proc.stderr}"
    out = {}
    for line in proc.stdout.splitlines():
        impl, off, ln, crc = line.split("\t")
        out[(impl, int(off), int(ln))] = int(crc, 16)
    assert out, "dumper produced no rows"
    return out


def _assert_matches_zlib(results, data, cases):
    impls = sorted({k[0] for k in results})
    assert impls == ["clmul", "slice8", "table"], f"unexpected impl set: {impls}"
    for off, ln in cases:
        want = zlib.crc32(data[off:off + ln])
        for impl in impls:
            got = results[(impl, off, ln)]
            assert got == want, (
                f"{impl} disagrees with zlib.crc32 at offset={off} len={ln}: "
                f"got 0x{got:08x}, want 0x{want:08x}"
            )


def test_selftest_vector_all_impls(kernel_bin):
    """The loader's own guard (hnsw.hpp) must hold for whichever impl is selected."""
    proc = subprocess.run([kernel_bin, "--selftest"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.count("PASS") == 3, proc.stdout
    assert "FAIL" not in proc.stdout


def test_every_length_0_to_256(kernel_bin, tmp_path):
    """Exhaustive over length. 96 is the real ex window; 9 is the loader self-test.

    A fold bug typically survives one length and dies on another, so sweeping every length is
    what makes this a proof rather than a spot check.
    """
    data = bytes((i * 7 + 13) % 256 for i in range(600))
    cases = [(0, n) for n in range(257)]
    _assert_matches_zlib(_run_cases(kernel_bin, tmp_path, data, cases), data, cases)


def test_fold_width_boundaries(kernel_bin, tmp_path):
    """Lengths straddling the 16 B fold width and the 8 B slice8 block."""
    import random
    rnd = random.Random(20260803)
    data = bytes(rnd.getrandbits(8) for _ in range(1024))
    lens = [0, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 47, 48,
            63, 64, 65, 95, 96, 97, 111, 112, 127, 128, 129, 255, 256, 512]
    cases = [(0, n) for n in lens]
    _assert_matches_zlib(_run_cases(kernel_bin, tmp_path, data, cases), data, cases)


def test_unaligned_offsets_at_the_real_window_size(kernel_bin, tmp_path):
    """96 B windows at every start offset mod 64.

    The ex window lives at element offset 168 with a 272 B stride, so it is neither 16- nor
    64-byte aligned in general. _mm_loadu_si128 is unaligned-safe, but this pins that claim.
    """
    import random
    rnd = random.Random(4242)
    data = bytes(rnd.getrandbits(8) for _ in range(4096))
    cases = [(off, 96) for off in range(0, 64)] + [(168 + 272 * i, 96) for i in range(8)]
    _assert_matches_zlib(_run_cases(kernel_bin, tmp_path, data, cases), data, cases)


def test_adversarial_patterns(kernel_bin, tmp_path):
    """All-zero and all-ones blow up sign/carry mistakes that random data hides."""
    for pattern in (b"\x00" * 512, b"\xff" * 512, b"\x00\xff" * 256, b"\x80" * 512):
        cases = [(0, n) for n in (0, 1, 15, 16, 17, 32, 96, 128, 256, 512)]
        _assert_matches_zlib(_run_cases(kernel_bin, tmp_path, pattern, cases), pattern, cases)


def test_impls_agree_with_each_other(kernel_bin, tmp_path):
    """Redundant with the zlib checks, but states the property the swap depends on directly."""
    import random
    rnd = random.Random(99)
    data = bytes(rnd.getrandbits(8) for _ in range(8192))
    cases = [(rnd.randrange(0, 4096), rnd.randrange(0, 4096)) for _ in range(200)]
    res = _run_cases(kernel_bin, tmp_path, data, cases)
    for off, ln in cases:
        vals = {impl: res[(impl, off, ln)] for impl in ("table", "slice8", "clmul")}
        assert len(set(vals.values())) == 1, f"impls disagree at {off}+{ln}: {vals}"


def test_header_declares_pclmul_guard():
    """The clmul path must be compile-time guarded, matching rabitqlib house style.

    rabitqlib uses `#if defined(__AVX512F__)` etc. and has zero runtime cpuid dispatch outside
    third/; a runtime check here would also put a branch on the query path.
    """
    text = open(HDR).read()
    assert "#if defined(__PCLMUL__)" in text
    # Strip // comments: the header discusses runtime dispatch in prose precisely to explain
    # why it does not use it, and a naive grep would flag that explanation as the offence.
    code = "\n".join(line.split("//", 1)[0] for line in text.splitlines())
    assert "__builtin_cpu_supports" not in code, "runtime dispatch is against house style"
    assert "cpuid" not in code.lower()
