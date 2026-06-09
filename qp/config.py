"""Frozen controlled variables for the whole study.

These are the knobs that prompt.txt / CLAUDE.md require to stay LOCKED across every
phase so that "algorithm differences" are never miscounted as "quantization differences".
Import from here; do not re-declare these constants elsewhere.
"""
import os

# --- paths --------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # quant_protection/
SIFT_DIR = os.path.join(ROOT, "sift")
SIFT_BASE = os.path.join(SIFT_DIR, "sift_base.fvecs")
SIFT_QUERY = os.path.join(SIFT_DIR, "sift_query.fvecs")
SIFT_GT = os.path.join(SIFT_DIR, "sift_groundtruth.ivecs")
SIFT_LEARN = os.path.join(SIFT_DIR, "sift_learn.fvecs")

DIM = 128

# --- IVF family (shared) ------------------------------------------------------
NLIST = 1024                       # inverted lists; shared by IVF_FLAT / SQ8 / PQ
NPROBE_GRID = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]

# --- graph family (shared) ----------------------------------------------------
M = 32                             # bidirectional connections per node
EF_CONSTRUCTION = 200
EF_SEARCH_GRID = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]

# --- PQ -----------------------------------------------------------------------
PQ_M = [8, 16]                     # sub-quantizers; nbits fixed at 8
PQ_NBITS = 8

# --- measurement --------------------------------------------------------------
K = 10                             # operating-point metric; also report @1 and @100
RECALL_KS = (1, 10, 100)
OPERATING_POINT = 0.95             # align every approximate index's clean recall@10 here
TARGET_BAND = (0.945, 0.955)       # "aligned" window for reporting
EPSILON_TOLERANT = 0.02            # tolerant recall: within (1+eps) of kth GT distance

SEED = 1234

DATASET = "SIFT1M"
FAISS_REFINE = False               # re-rank OFF by default (would rewrite the corruption story)

# --- design matrix (names are the artifact filenames too) ---------------------
# kind: "exact" (FLAT, no tuning) | "ivf" (knob=nprobe) | "graph" (knob=efSearch)
INDEX_SPECS = [
    {"name": "FLAT",       "kind": "exact", "trained": False, "quant": None},
    {"name": "IVF_FLAT",   "kind": "ivf",   "trained": True,  "quant": None},
    {"name": "IVF_SQ8",    "kind": "ivf",   "trained": True,  "quant": "sq8"},
    {"name": "IVF_PQ_M8",  "kind": "ivf",   "trained": True,  "quant": "pq", "pq_m": 8},
    {"name": "IVF_PQ_M16", "kind": "ivf",   "trained": True,  "quant": "pq", "pq_m": 16},
    {"name": "HNSW",       "kind": "graph", "trained": False, "quant": None},
    {"name": "HNSW_SQ8",   "kind": "graph", "trained": True,  "quant": "sq8"},
]

# --- Phase 2 (ADDITIVE — new defaults only; never change the locked values above) -----
# These are Phase 2 knobs. They do not alter any controlled variable from Phase 0/1; they
# only parameterize the rate rollup, the PQ collapse threshold, the burst sweep, and the
# subprocess-isolation timeout. Override per-run via CLI flags on the phase2_* scripts.

# Per-bit error-rate axis for the Tier 3 faults/MB rollup. Absolute per-bit probability —
# physically meaningful (a DRAM field study delivers a per-bit rate; collapse risk is then
# driven by the ABSOLUTE bit count of a critical structure, not a per-flip conditional prob).
# README converts these to/from DRAM FIT/Mbit (e.g. Schroeder et al. field study).
PHASE2_RATE_GRID = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

# PQ has a low own-baseline (M8≈0.379, M16≈0.563) so its "catastrophic" is RELATIVE:
# a flip is catastrophic if faulted recall@10 retains < this fraction of the index's own
# clean recall@10. Aligned indexes keep the absolute buckets.CATASTROPHIC_ABS (>0.01).
PHASE2_PQ_RETENTION_FRAC = 0.5

# Burst length grid in BITS, spanning sq_scale (8192 bits) up to pq_codebook (~1.05M bits)
# and beyond — to expose how spatially-clustered errors (a bad DIMM block) magnify with the
# absolute size of a critical structure.
PHASE2_BURST_B = [1, 8, 64, 512, 1024, 8192, 65536, 262144]

# Per-flip subprocess wall-clock budget (seconds) for the isolated graph_edges / burst
# injectors. A child still alive past this is terminated and recorded as a `crash` (hang).
PHASE2_FLIP_TIMEOUT_S = 120.0
