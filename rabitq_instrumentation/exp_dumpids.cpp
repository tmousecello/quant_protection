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
// Usage: exp_dumpids <index> <query.fvecs> <gt.ivecs> <l2|ip> <ef> <out.ivecs> [topk=10]
//                    [--recovery none|drop|fallback_eb] [--crc-manifest <path>]
//                    [--stats-json <path>]
//   stdout: "RECALL\t<recall@topk>" (C++-side recall on the dumped ids, for parity)
//   out.ivecs: nq rows, each = [uint32 topk][topk * uint32 id]; matches qp.data.read_ivecs.
//              short rows (possible under --recovery drop at high corruption) are padded with
//              0xFFFFFFFF (-1), which qp.metrics treats as a non-matching id.
//   stats json (when --stats-json): load-time CRC scan totals + per-query recovery counters
//              (consults / corrupt_hits / fallbacks / drops).
// Determinism: same index file + same manifest + same flags -> identical ids (single-thread
// search, no RNG anywhere on this path).
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
              << " [--stats-json <path>]\n";
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
    for (; argi < argc; argi++) {
        std::string a(argv[argi]);
        if (a == "--recovery" && argi + 1 < argc) {
            recovery = argv[++argi];
        } else if (a == "--crc-manifest" && argi + 1 < argc) {
            manifest_path = argv[++argi];
        } else if (a == "--stats-json" && argi + 1 < argc) {
            stats_path = argv[++argi];
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

    if (recovery == "drop") {
        hnsw.load_crc_manifest(manifest_path.c_str(), index_type::FAULT_DROP);
    } else if (recovery == "fallback_eb") {
        hnsw.load_crc_manifest(manifest_path.c_str(), index_type::FAULT_FALLBACK_EB);
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
    for (size_t q = 0; q < nq; q++) {
        hnsw.fi_reset_query_stats();
        auto one = hnsw.search(query.data() + (q * dim), 1, topk, ef, 1);
        res[q] = std::move(one[0]);
        const auto& s = hnsw.fi_qstats_;
        per_query.push_back({s.consults, s.corrupt_hits, s.fallbacks, s.drops});
        tot_consults += s.consults;
        tot_corrupt += s.corrupt_hits;
        tot_fallbacks += s.fallbacks;
        tot_drops += s.drops;
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
           << "  \"load\": {\"elements_checked\": " << hnsw.fi_crc_checked_
           << ", \"elements_crc_fail\": " << hnsw.fi_crc_failed_ << "},\n"
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
