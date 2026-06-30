"""Samuel's EB-aware fallback ranking policy — direct port of hnsw.hpp:1313-1319.

The recovery logic lives in C++ (StorageSystemProject_RaBitQ_Free-recovery,
third_party/RaBitQ-Library/include/rabitqlib/index/hnsw/hnsw.hpp, lines 1313-1319).
This module is a mechanical transcription of that formula into Python so the Phase 3
Stage-2 slope-layer can apply per-vector EB ranking without re-implementing any policy.
"""


def eb_rank_dist(est_dist, f_error, g_error):
    """Pessimistic upper-bound rank distance for a vector whose ex_data is corrupted.

    Mirrors FAULT_FALLBACK_EB in hnsw.hpp:
      low_dist  = est_dist - f_error * g_error   (bin-only lower bound)
      rank_dist = est_dist + (est_dist - low_dist) == est_dist + f_error * g_error

    A corrupted vector's rank distance is inflated above its 1-bit bin estimate,
    preventing it from evicting a true neighbour via a falsely precise ex distance.
    Returns a scalar (float or numpy scalar, matching input type).
    """
    low_dist = est_dist - f_error * g_error
    return est_dist + (est_dist - low_dist)
