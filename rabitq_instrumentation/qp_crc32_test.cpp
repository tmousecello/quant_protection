// qp_crc32_test.cpp — correctness dumper and microbenchmark for qp_crc32.hpp.
//
// Two jobs, deliberately in one binary so the thing benchmarked is the thing proven correct.
//
//   --dump <datafile> <casefile>
//       Reads a blob and a list of "<offset> <len>" cases, prints one TSV row per
//       (impl, case): "<impl>\t<offset>\t<len>\t<crc_hex>".  Python owns the data and the case
//       list, so there is no PRNG to reimplement on both sides and no way for the two to drift.
//       artifacts/phase3/tests/test_crc_kernel.py compares every row against zlib.crc32.
//
//   --bench [--scattered]
//       Hot-buffer throughput per impl at 9 / 96 / 1024 / 96*2^20 bytes.  96 B is the size that
//       decides anything here: it is the ex window on SIFT b=7.  The hot buffer isolates KERNEL
//       COMPUTE (everything in L1); --scattered instead walks 96 B windows over a 280 MB region
//       in a shuffled order, which is the access pattern `drop` actually has, and is where the
//       memory-traffic ceiling shows up.
//
//   --selftest
//       The loader's own vector, for every impl.
//
// Builds two ways on purpose: standalone (`g++ -O2 -march=native qp_crc32_test.cpp`) so
// bit-identity is testable on any machine without the RaBitQ tree or the 280 MB index, and as a
// CMake target next to exp_dumpids so the benchmark runs under the project's real flags.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "qp_crc32.hpp"

namespace {

const int kImpls[] = {qp_crc::kTable, qp_crc::kSlice8, qp_crc::kClmul};

std::vector<unsigned char> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(2);
    }
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<unsigned char> buf(static_cast<size_t>(n));
    if (n > 0) f.read(reinterpret_cast<char*>(buf.data()), n);
    return buf;
}

int cmd_dump(const char* datafile, const char* casefile) {
    const std::vector<unsigned char> data = read_file(datafile);
    std::ifstream cf(casefile);
    if (!cf) {
        std::fprintf(stderr, "cannot open %s\n", casefile);
        return 2;
    }
    size_t off = 0, len = 0;
    while (cf >> off >> len) {
        if (off + len > data.size()) {
            std::fprintf(stderr, "case out of range: %zu+%zu > %zu\n", off, len, data.size());
            return 2;
        }
        for (int impl : kImpls) {
            const uint32_t c = qp_crc::crc32(impl, data.data() + off, len);
            std::printf("%s\t%zu\t%zu\t%08x\n", qp_crc::impl_name(impl), off, len, c);
        }
    }
    return 0;
}

// Hot-buffer: the whole input stays in L1/L2, so this is kernel compute and nothing else.
void bench_hot(size_t nbytes, size_t iters) {
    std::vector<unsigned char> buf(nbytes);
    std::mt19937_64 rng(12345);
    for (auto& b : buf) b = static_cast<unsigned char>(rng());

    std::printf("  %-10zu B", nbytes);
    for (int impl : kImpls) {
        // One untimed pass so tables are built and the buffer is warm for every impl equally.
        volatile uint32_t sink = qp_crc::crc32(impl, buf.data(), nbytes);
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iters; i++) {
            sink = qp_crc::crc32(impl, buf.data(), nbytes);
        }
        const auto t1 = std::chrono::steady_clock::now();
        (void)sink;
        const double ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        const double per_call = ns / static_cast<double>(iters);
        std::printf("   %-7s %8.1f ns  %6.3f ns/B", qp_crc::impl_name(impl), per_call,
                    per_call / static_cast<double>(nbytes));
    }
    std::printf("\n");
}

// Scattered: 96 B windows drawn in shuffled order from a region the size of the real level0
// block, so most reads miss cache. This is `drop`'s actual pattern -- 20 634 unrelated
// neighbours per query -- and is the arm where memory, not the kernel, sets the floor.
void bench_scattered(size_t region_bytes, size_t window, size_t nwindows) {
    std::vector<unsigned char> buf(region_bytes);
    std::mt19937_64 rng(6789);
    for (auto& b : buf) b = static_cast<unsigned char>(rng());

    // Element-aligned offsets, shuffled, mirroring how ids arrive from the graph.
    const size_t stride = 272;  // size_data_per_element on this index
    std::vector<size_t> offs;
    offs.reserve(nwindows);
    const size_t nelem = (region_bytes - window) / stride;
    for (size_t i = 0; i < nwindows; i++) offs.push_back((i % nelem) * stride + 168);
    std::shuffle(offs.begin(), offs.end(), rng);

    std::printf("  scattered %zu B x %zu over %zu MB", window, nwindows, region_bytes >> 20);
    for (int impl : kImpls) {
        volatile uint32_t sink = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t o : offs) sink = qp_crc::crc32(impl, buf.data() + o, window);
        const auto t1 = std::chrono::steady_clock::now();
        (void)sink;
        const double ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        const double per_call = ns / static_cast<double>(nwindows);
        std::printf("   %-7s %8.1f ns  %6.3f ns/B", qp_crc::impl_name(impl), per_call,
                    per_call / static_cast<double>(window));
    }
    std::printf("\n");
}

int cmd_bench(bool scattered) {
    std::printf("pclmul_compiled_in=%d\n", qp_crc::impl_available(qp_crc::kClmul) ? 1 : 0);
    std::printf("hot buffer (kernel compute only):\n");
    bench_hot(9, 2000000);
    bench_hot(96, 2000000);
    bench_hot(1024, 200000);
    bench_hot(96u << 20, 3);
    if (scattered) {
        std::printf("scattered (memory-bound, mirrors drop's access pattern):\n");
        bench_scattered(280u << 20, 96, 2000000);
    }
    return 0;
}

int cmd_selftest() {
    int rc = 0;
    for (int impl : kImpls) {
        const bool ok = qp_crc::self_test(impl);
        std::printf("%-7s selftest %s\n", qp_crc::impl_name(impl), ok ? "PASS" : "FAIL");
        if (!ok) rc = 1;
    }
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--selftest") == 0) return cmd_selftest();
    if (argc == 4 && std::strcmp(argv[1], "--dump") == 0) return cmd_dump(argv[2], argv[3]);
    if (argc >= 2 && std::strcmp(argv[1], "--bench") == 0) {
        const bool scattered = (argc >= 3 && std::strcmp(argv[2], "--scattered") == 0);
        return cmd_bench(scattered);
    }
    std::fprintf(stderr,
                 "Usage: %s --dump <datafile> <casefile>\n"
                 "       %s --bench [--scattered]\n"
                 "       %s --selftest\n",
                 argv[0], argv[0], argv[0]);
    return 1;
}
