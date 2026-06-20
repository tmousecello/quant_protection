// exp_dumpids.cpp — Phase 3 Stage 0 parity instrument (quant_protection).
//
// Dumps the REAL RaBitQ search path's per-query top-k neighbour ids so the Python side can
// recompute recall with qp.metrics and check it equals this binary's recall on the SAME ids
// (the keystone qp-vs-C++ parity). Mirrors sample/hnsw_rabitq_querying.cpp exactly (same
// includes, types, and hnsw.search call) but runs a single ef and writes ids instead of a
// recall sweep.
//
// Usage: exp_dumpids <index> <query.fvecs> <gt.ivecs> <l2|ip> <ef> <out.ivecs> [topk=10]
//   stdout: "RECALL\t<recall@topk>" (C++-side recall on the dumped ids, for parity)
//   out.ivecs: nq rows, each = [uint32 topk][topk * uint32 id]; matches qp.data.read_ivecs.
//              short rows (shouldn't happen for a full index) are padded with 0xFFFFFFFF (-1),
//              which qp.metrics treats as a non-matching id.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/utils/io.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::hnsw::HierarchicalNSW;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <index> <query.fvecs> <gt.ivecs> <l2|ip> <ef> <out.ivecs> [topk=10]\n";
        return 1;
    }

    const char* index_file = argv[1];
    const char* query_file = argv[2];
    const char* gt_file = argv[3];
    std::string metric_str(argv[4]);
    const size_t ef = static_cast<size_t>(std::stoul(argv[5]));
    const char* out_file = argv[6];
    const size_t topk = (argc > 7) ? static_cast<size_t>(std::stoul(argv[7])) : 10;

    rabitqlib::MetricType metric_type =
        (metric_str == "ip" || metric_str == "IP") ? rabitqlib::METRIC_IP : rabitqlib::METRIC_L2;

    data_type query;
    gt_type gt;
    rabitqlib::load_vecs<float, data_type>(query_file, query);
    rabitqlib::load_vecs<uint32_t, gt_type>(gt_file, gt);
    const size_t nq = query.rows();

    index_type hnsw;
    hnsw.load(index_file, metric_type);

    // Same call/shape as the querying sample: per-query vector of (dist, id) pairs.
    std::vector<std::vector<std::pair<float, PID>>> res =
        hnsw.search(query.data(), nq, topk, ef, 1);

    // Write ivecs: fixed topk-width rows so qp.data.read_ivecs reshapes cleanly.
    std::ofstream out(out_file, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Cannot open output file " << out_file << '\n';
        return 1;
    }
    const uint32_t dim = static_cast<uint32_t>(topk);
    const uint32_t kPad = 0xFFFFFFFFU;  // == int32 -1: a non-matching id for recall
    size_t total_correct = 0;
    for (size_t i = 0; i < nq; i++) {
        out.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
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
    return 0;
}
