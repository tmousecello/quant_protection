"""Design-matrix index builders + clean-recall alignment.

One builder per index type, all using the shared parameters from `config` so the only
thing that varies across the matrix is the ENCODING (the study's independent variable).
Trained indexes train on sift_learn (100k, disjoint from base) to avoid train/add overlap.

`tune` aligns each approximate index's clean recall@10 to the common operating point by
sweeping its single search knob (nprobe for IVF, efSearch for graph) and picking the
smallest setting that reaches the target. FLAT is exact and is never tuned.
"""
import faiss
import numpy as np

from . import config
from .metrics import recall_at_k


# --- construction -------------------------------------------------------------
def build_index(spec, xb, xl):
    """Build and populate one index from a config.INDEX_SPECS entry.

    xb: base vectors to add. xl: training vectors (sift_learn). Returns the ready index.
    """
    d = xb.shape[1]
    name = spec["name"]

    if name == "FLAT":
        idx = faiss.IndexFlatL2(d)
        idx.add(xb)
        return idx

    if spec["kind"] == "ivf":
        quantizer = faiss.IndexFlatL2(d)
        if spec["quant"] is None:
            idx = faiss.IndexIVFFlat(quantizer, d, config.NLIST)
        elif spec["quant"] == "sq8":
            idx = faiss.IndexIVFScalarQuantizer(
                quantizer, d, config.NLIST,
                faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
        elif spec["quant"] == "pq":
            idx = faiss.IndexIVFPQ(
                quantizer, d, config.NLIST, spec["pq_m"], config.PQ_NBITS)
        else:
            raise ValueError(f"unknown ivf quant {spec['quant']}")
        idx.train(xl)
        idx.add(xb)
        return idx

    if spec["kind"] == "graph":
        if spec["quant"] is None:
            idx = faiss.IndexHNSWFlat(d, config.M)
            idx.hnsw.efConstruction = config.EF_CONSTRUCTION
            idx.add(xb)
        elif spec["quant"] == "sq8":
            idx = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, config.M)
            idx.hnsw.efConstruction = config.EF_CONSTRUCTION
            idx.train(xl)          # SQ storage needs trained vmin/vdiff before add
            idx.add(xb)
        else:
            raise ValueError(f"unknown graph quant {spec['quant']}")
        return idx

    raise ValueError(f"unknown index kind {spec['kind']}")


# --- search-knob plumbing -----------------------------------------------------
def knob_name(spec):
    return {"exact": None, "ivf": "nprobe", "graph": "efSearch"}[spec["kind"]]


def set_knob(index, spec, value):
    """Set the search-time knob for an index (nprobe / efSearch)."""
    if spec["kind"] == "ivf":
        index.nprobe = int(value)
    elif spec["kind"] == "graph":
        index.hnsw.efSearch = int(value)
    # exact: nothing to set


def knob_grid(spec):
    return {"ivf": config.NPROBE_GRID, "graph": config.EF_SEARCH_GRID}[spec["kind"]]


# --- clean-recall alignment ---------------------------------------------------
def _recall_at_knob(index, spec, value, xq, gt_ids, k):
    set_knob(index, spec, value)
    _, I = index.search(xq, k)
    return recall_at_k(I, gt_ids, k)


def tune(index, spec, xq, gt_ids, target=config.OPERATING_POINT, k=config.K):
    """Align clean recall@k to `target` by choosing the smallest knob value reaching it.

    A coarse ascending grid first brackets the crossing; then an integer binary search
    between the last-below and first-at-or-above grid points lands as close to `target`
    from above as the discrete knob allows (so the index sits near the band, not well past
    it). Returns (chosen_value, recall_at_chosen, curve, target_met). `curve` records every
    knob value actually evaluated (coarse + refine) so the alignment is auditable.

    If the index plateaus below target (e.g. the IVF_PQ codebook ceiling on SIFT, which is
    unreachable without re-ranking), returns the best setting with target_met=False — a
    documented fallback, not a failure.
    """
    grid = knob_grid(spec)
    curve = []
    prev = None                  # last grid value strictly below target
    bracket_hi = None            # first grid value at/above target
    best = (grid[0], -1.0)
    for value in grid:
        r = _recall_at_knob(index, spec, value, xq, gt_ids, k)
        curve.append([int(value), r])
        if r > best[1]:
            best = (value, r)
        if r >= target:
            bracket_hi = value
            break
        prev = value

    if bracket_hi is None:
        chosen, chosen_recall, target_met = best[0], best[1], False
    else:
        # refine: smallest integer knob in (prev, bracket_hi] with recall >= target.
        lo = (prev + 1) if prev is not None else 1
        hi = bracket_hi
        chosen, chosen_recall = bracket_hi, dict(curve)[bracket_hi]
        while lo < hi:
            mid = (lo + hi) // 2
            r = _recall_at_knob(index, spec, mid, xq, gt_ids, k)
            curve.append([int(mid), r])
            if r >= target:
                chosen, chosen_recall, hi = mid, r, mid
            else:
                lo = mid + 1
        target_met = True

    set_knob(index, spec, chosen)             # leave index pinned at the operating point
    return int(chosen), float(chosen_recall), curve, target_met
