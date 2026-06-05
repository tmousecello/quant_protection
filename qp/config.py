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
