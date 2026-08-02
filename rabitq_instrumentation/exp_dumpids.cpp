// exp_dumpids.cpp — Phase 3 Stage 0 parity instrument + Stage 2 Option B recovery harness
// (quant_protection).
//
// Dumps the REAL RaBitQ search path's per-query top-k neighbour ids so the Python side can
// recompute recall with qp.metrics and check it equals this binary's recall on the SAME ids
// (the keystone qp-vs-C++ parity). Mirrors sample/hnsw_rabitq_querying.cpp (same includes,
// types, and search call) but runs a single ef and writes ids instead of a recall sweep.
//
// Option B (--recovery): query-time corruption recovery driven by a REAL CRC check against a
// Python-written manifest (qp.rabitq.crc_manifest — written from the CLEAN index BEFORE
// injection). This binary NEVER injects faults and never defines "clean"; it only loads a
// (possibly corrupted) index, compares its bytes to the manifest CRCs (hnsw.hpp
// load_crc_manifest), and applies Samuel's exp-3 fault policies during search:
//   none        : no CRC check at all; possibly-corrupted ex codes are used as-is.
//   drop        : CRC-failed element -> candidate skipped entirely ("detect & skip").
//   fallback_eb : CRC-failed element -> ranked by the pessimistic error-bound distance
//                 est + (est - low) from the intact bin code + f_error (hnsw.hpp, exp-3
//                 semantics, unchanged).
// The stdout RECALL line is for adapter parity only — the AUTHORITATIVE recall is recomputed
// in Python by qp.metrics from the dumped ids (hard rule #2).
//
//
// --crc-mode selects WHEN the CRC is computed, not what it decides:
//   load : eager whole-index scan at load (the original behaviour, still the default).
//   lazy : on-access — recompute an element's CRC at the moment the search consults it.
//          This is the mode the paper's Sec 3.1/3.3 describes; it drops the "index memory is
//          immutable while queries run" assumption that makes the eager scan equivalent.
// Both modes use the same predicate at the same decision points, so on an index that does not
// change during the run they must return bit-identical ids — that is an acceptance gate
// (phase3_expb_recovery.py --lazy-gate), not an assumption.
//
// Usage: exp_dumpids <index> <query.fvecs> <gt.ivecs> <l2|ip> <ef> <out.ivecs> [topk=10]
//                    [--recovery none|drop|fallback_eb] [--crc-manifest <path>]
//                    [--stats-json <path>] [--crc-mode load|lazy] [--crc-timer]
//   stdout: "RECALL\t<recall@topk>" (C++-side recall on the dumped ids, for parity)
//   out.ivecs: nq rows, each = [uint32 topk][topk * uint32 id]; matches qp.data.read_ivecs.
//              short rows (possible under --recovery drop at high corruption) are padded with
//              0xFFFFFFFF (-1), which qp.metrics treats as a non-matching id.
//   stats json (when --stats-json): crc_mode + load-time CRC scan totals + per-query recovery
//              counters (consults / corrupt_hits / fallbacks / drops) + the "crc" block
//              (on-access check/byte/ns totals, lazy mode only) + the two E7-microbench timers,
//              load.crc_scan_ns (the eager CRC loop, hnsw.hpp) and search_wall_ns (this file:
//              the per-query hnsw.search calls only, no IO/recall/JSON). Dividing each by its
//              own count -- crc_scan_ns/load.elements_checked vs search_wall_ns/
//              totals.consults -- gives the per-vector eager-scan cost vs the per-candidate
//              search cost. Mind the DIRECTION of the bias: the denominator bundles graph
//              traversal in with distance work, which inflates it, so the measured ratio is a
//              LOWER bound on the true CRC-to-pure-distance ratio. The true ratio is >= the
//              measured one, i.e. CRC is AT MOST that much cheaper and possibly far less so:
//              an AVX2 128-d L2 kernel is ~20-40 ns against ~110 ns/vector of eager scan, so
//              against ONE pure distance the CRC may well be the more expensive of the two.
//              The framing to quote, which needs no such caveat, is the aggregate: the
//              one-time eager CRC scan costs ~110.7 ms against 15.64 s of search for 10K
//              queries at ef=2000 -- 0.71% of search time.
// Determinism: same index file + same manifest + same flags -> identical ids (single-thread
// search, no RNG anywhere on this path).
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/utils/io.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::hnsw::HierarchicalNSW;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <index> <query.fvecs> <gt.ivecs> <l2|ip> <ef> <out.ivecs> [topk=10]\n"
              << "       [--recovery none|drop|fallback_eb] [--crc-manifest <path>]"
              << " [--stats-json <path>]\n"
              << "       [--crc-mode load|lazy] [--crc-timer]"
              << " [--crc-impl table|slice8|clmul]\n";
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        usage(argv[0]);
        return 1;
    }

    const char* index_file = argv[1];
    const char* query_file = argv[2];
    const char* gt_file = argv[3];
    std::string metric_str(argv[4]);
    const size_t ef = static_cast<size_t>(std::stoul(argv[5]));
    const char* out_file = argv[6];

    // Optional positional topk (argv[7] iff it is not a flag), then trailing --key value
    // flags. Positional args 1-6 are kept exactly as Stage 0 shipped them so the existing
    // adapter.query_ids call keeps working unchanged.
    size_t topk = 10;
    int argi = 7;
    if (argi < argc && std::strncmp(argv[argi], "--", 2) != 0) {
        topk = static_cast<size_t>(std::stoul(argv[argi]));
        argi++;
    }
    std::string recovery = "none";
    std::string manifest_path;
    std::string stats_path;
    std::string crc_mode = "load";
    std::string crc_impl = "table";
    bool crc_timer = false;
    for (; argi < argc; argi++) {
        std::string a(argv[argi]);
        if (a == "--recovery" && argi + 1 < argc) {
            recovery = argv[++argi];
        } else if (a == "--crc-manifest" && argi + 1 < argc) {
            manifest_path = argv[++argi];
        } else if (a == "--stats-json" && argi + 1 < argc) {
            stats_path = argv[++argi];
        } else if (a == "--crc-mode" && argi + 1 < argc) {
            crc_mode = argv[++argi];
        } else if (a == "--crc-impl" && argi + 1 < argc) {
            crc_impl = argv[++argi];
        } else if (a == "--crc-timer") {
            crc_timer = true;
        } else {
            std::cerr << "unknown/incomplete arg: " << a << '\n';
            usage(argv[0]);
            return 1;
        }
    }
    if (recovery != "none" && recovery != "drop" && recovery != "fallback_eb") {
        std::cerr << "bad --recovery " << recovery << '\n';
        usage(argv[0]);
        return 1;
    }
    if (recovery != "none" && manifest_path.empty()) {
        std::cerr << "--recovery " << recovery << " requires --crc-manifest\n";
        usage(argv[0]);
        return 1;
    }
    if (crc_mode != "load" && crc_mode != "lazy") {
        std::cerr << "bad --crc-mode " << crc_mode << " (expected load|lazy)\n";
        usage(argv[0]);
        return 1;
    }
    // A mistyped --crc-impl must report-and-stop, never silently fall back to `table` -- that
    // would quietly benchmark the wrong kernel and report the name you asked for.
    const int crc_impl_id = qp_crc::impl_from_name(crc_impl.c_str());
    if (crc_impl_id < 0) {
        std::cerr << "bad --crc-impl " << crc_impl << " (expected table|slice8|clmul)\n";
        usage(argv[0]);
        return 1;
    }
    if (!qp_crc::impl_available(crc_impl_id)) {
        std::cerr << "--crc-impl " << crc_impl << " was not compiled in (missing __PCLMUL__); "
                  << "rebuild with -march=native\n";
        return 1;
    }
    // --recovery none loads no manifest and checks no CRC at all, so "when" and "how" are both
    // meaningless there. Reject rather than silently reporting a crc_mode/crc_impl on a run
    // that never CRC'd a single byte.
    if (recovery == "none" && (crc_mode != "load" || crc_timer || crc_impl != "table")) {
        std::cerr << "--recovery none never checks a CRC; "
                  << "--crc-mode/--crc-timer/--crc-impl do not apply\n";
        usage(argv[0]);
        return 1;
    }

    rabitqlib::MetricType metric_type =
        (metric_str == "ip" || metric_str == "IP") ? rabitqlib::METRIC_IP : rabitqlib::METRIC_L2;

    data_type query;
    gt_type gt;
    rabitqlib::load_vecs<float, data_type>(query_file, query);
    rabitqlib::load_vecs<uint32_t, gt_type>(gt_file, gt);
    const size_t nq = query.rows();
    const size_t dim = query.cols();

    index_type hnsw;
    hnsw.load(index_file, metric_type);

    const bool crc_lazy = (crc_mode == "lazy");
    hnsw.fi_crc_timer_ = crc_timer;
    // Must be set BEFORE load_crc_manifest: the loader self-tests the selected kernel, and the
    // eager scan uses it for all 10^6 elements. Assigning a member that only the updated
    // hnsw.hpp has is also the tripwire for a stale patch -- if build_rabitq.sh's sentinel was
    // not bumped, this line fails to COMPILE rather than silently running the old kernel while
    // the stats JSON reports the impl you asked for.
    hnsw.fi_crc_impl_ = crc_impl_id;
    if (recovery == "drop") {
        hnsw.load_crc_manifest(manifest_path.c_str(), index_type::FAULT_DROP, crc_lazy);
    } else if (recovery == "fallback_eb") {
        hnsw.load_crc_manifest(manifest_path.c_str(), index_type::FAULT_FALLBACK_EB, crc_lazy);
    }
    // recovery == "none": no manifest load, fault_policy_ stays FAULT_NONE — no CRC check,
    // the (possibly corrupted) ex codes are consumed exactly as Stage 0 did.

    // One search call per query (nq=1, single thread) so per-query recovery counters can be
    // read/reset between queries. Bit-identical to the old single hnsw.search(nq) call:
    // parallel_for with numThreads==1 is a plain sequential loop and all per-query state
    // (rotated query, estimators) is local to the call.
    std::vector<std::vector<std::pair<float, PID>>> res(nq);
    std::vector<std::vector<uint64_t>> per_query;
    per_query.reserve(nq);
    uint64_t tot_consults = 0;
    uint64_t tot_corrupt = 0;
    uint64_t tot_fallbacks = 0;
    uint64_t tot_drops = 0;
    // On-access CRC totals (all stay 0 under --crc-mode load: the query path computes no CRC).
    uint64_t tot_crc_checks = 0;
    uint64_t tot_crc_bytes = 0;
    uint64_t tot_crc_ns = 0;
    uint64_t tot_crc_oob = 0;
    // Search-only wall time: the clock brackets the hnsw.search call and nothing else (no
    // counter reads, no result move, no IO), accumulated over all nq queries. Two
    // steady_clock reads per query against a ~ms-scale ef=2000 search is noise-level
    // overhead, and it is identical work on every run -- the dumped ids cannot change.
    uint64_t search_wall_ns = 0;
    for (size_t q = 0; q < nq; q++) {
        hnsw.fi_reset_query_stats();
        const auto q_t0 = std::chrono::steady_clock::now();
        auto one = hnsw.search(query.data() + (q * dim), 1, topk, ef, 1);
        search_wall_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - q_t0)
                .count());
        res[q] = std::move(one[0]);
        const auto& s = hnsw.fi_qstats_;
        per_query.push_back({s.consults, s.corrupt_hits, s.fallbacks, s.drops});
        tot_consults += s.consults;
        tot_corrupt += s.corrupt_hits;
        tot_fallbacks += s.fallbacks;
        tot_drops += s.drops;
        tot_crc_checks += s.crc_checks;
        tot_crc_bytes += s.crc_bytes;
        tot_crc_ns += s.crc_ns;
        tot_crc_oob += s.crc_oob_skipped;
    }

    // Write ivecs: fixed topk-width rows so qp.data.read_ivecs reshapes cleanly.
    std::ofstream out(out_file, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Cannot open output file " << out_file << '\n';
        return 1;
    }
    const uint32_t dim_field = static_cast<uint32_t>(topk);
    const uint32_t kPad = 0xFFFFFFFFU;  // == int32 -1: a non-matching id for recall
    size_t total_correct = 0;
    for (size_t i = 0; i < nq; i++) {
        out.write(reinterpret_cast<const char*>(&dim_field), sizeof(uint32_t));
        for (size_t j = 0; j < topk; j++) {
            uint32_t id = (j < res[i].size()) ? static_cast<uint32_t>(res[i][j].second) : kPad;
            out.write(reinterpret_cast<const char*>(&id), sizeof(uint32_t));
        }
        // C++-side recall on these exact ids (per-pred-slot, any-gt-hit), same as the sample.
        for (size_t j = 0; j < topk && j < res[i].size(); j++) {
            for (size_t k = 0; k < topk; k++) {
                if (gt(i, k) == res[i][j].second) {
                    total_correct++;
                    break;
                }
            }
        }
    }
    out.close();

    const double recall = static_cast<double>(total_correct) /
                          static_cast<double>(nq * topk);
    std::cout << "RECALL\t" << recall << '\n';
    std::cout << "wrote " << nq << " x " << topk << " ids -> " << out_file << '\n';

    if (!stats_path.empty()) {
        std::ofstream sj(stats_path);
        if (!sj.is_open()) {
            std::cerr << "Cannot open stats file " << stats_path << '\n';
            return 1;
        }
        sj << "{\n"
           << "  \"recovery\": \"" << recovery << "\",\n"
           << "  \"crc_manifest\": \"" << manifest_path << "\",\n"
           << "  \"ef\": " << ef << ",\n"
           << "  \"topk\": " << topk << ",\n"
           << "  \"nq\": " << nq << ",\n"
           << "  \"crc_mode\": \"" << crc_mode << "\",\n"
           // Echoed from the id the index was actually configured with, not from the argv
           // string, so a run can never report a kernel it did not use.
           << "  \"crc_impl\": \"" << qp_crc::impl_name(hnsw.fi_crc_impl_) << "\",\n"
           // The load.* trio is all zero under --recovery none (no manifest is loaded, so
           // there is nothing to check and no scan to time) AND under --crc-mode lazy (no
           // scan ran). Consumers that read elements_crc_fail as a whole-index corruption
           // count -- phase3_g_detection, phase3_e7_episode -- must gate on crc_mode.
           << "  \"load\": {\"elements_checked\": " << hnsw.fi_crc_checked_
           << ", \"elements_crc_fail\": " << hnsw.fi_crc_failed_
           << ", \"crc_scan_ns\": " << hnsw.fi_crc_scan_ns_ << "},\n"
           // On-access detection totals, --crc-mode lazy only. `ns` is 0 unless --crc-timer:
           // two steady_clock reads around a ~96-byte software CRC inflate it by tens of
           // percent AND perturb search_wall_ns, which is the unperturbed load-vs-lazy delta
           // the acceptance run quotes. distinct_elements_failed counts elements that ever
           // failed a check (a stats-only set, never consulted by a decision), so it is
           // directly comparable to load mode's elements_crc_fail; checks/bytes count every
           // recomputation because the cost being measured is the recomputation.
           << "  \"crc\": {\"checks\": " << tot_crc_checks
           << ", \"bytes\": " << tot_crc_bytes
           << ", \"ns\": " << tot_crc_ns
           << ", \"oob_skipped\": " << tot_crc_oob
           << ", \"timer_enabled\": " << (crc_timer ? "true" : "false")
           << ", \"distinct_elements_failed\": " << hnsw.fi_crc_distinct_failed_ << "},\n"
           << "  \"search_wall_ns\": " << search_wall_ns << ",\n"
           << "  \"totals\": {\"consults\": " << tot_consults
           << ", \"corrupt_hits\": " << tot_corrupt
           << ", \"fallbacks\": " << tot_fallbacks
           << ", \"drops\": " << tot_drops << "},\n"
           << "  \"per_query\": [";
        for (size_t i = 0; i < per_query.size(); i++) {
            sj << (i == 0 ? "" : ",") << '[' << per_query[i][0] << ',' << per_query[i][1]
               << ',' << per_query[i][2] << ',' << per_query[i][3] << ']';
        }
        sj << "]\n}\n";
        sj.close();
    }
    return 0;
}
