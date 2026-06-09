#!/usr/bin/env python3
"""Phase 2 — Tier 3: faults/MB rollup (pure analysis — no bit flips).

Synthesizes the study's two headline curves from the single-bit maps already produced, by
sweeping an ABSOLUTE per-bit error rate r (config.PHASE2_RATE_GRID; convertible to a DRAM
field-study FIT/Mbit rate — see README). Normalizing by an absolute per-bit rate is the
physically meaningful choice: hardware delivers a per-bit rate, and collapse risk is then
driven by the ABSOLUTE bit count of a critical structure, not a per-flip conditional prob.

  Curve A — expected ΔRecall(r):   per index, Σ_region (region_bits · r) · E[ΔR@10 | flip].
            Bulk codes/vectors ≈ 0, centroid mild, the metadata tail rare-but-large. Linear
            superposition (valid while region_bits·r ≪ 1).
  Curve B — P(catastrophic collapse)(r):  per index, 1 − Π_region (1 − p_fatal_region),
            with p_fatal_region = 1 − exp(−cat_bits_region · r) and
            cat_bits_region = catastrophic_fraction · region_bits. fp32 indexes (FLAT /
            IVF_FLAT / HNSW) have no catastrophic single-bit structure ⇒ Curve B ≈ 0; SQ8 is
            driven by `sq_scale`, PQ by `pq_codebook`. THIS is the "quantization introduces a
            catastrophic single-point structure that fp32 lacks" main-axis plot data.

Per-tag map rows are collapsed to per-region quantities by BIT-SHARE weighting the tags
(fp32: sign 1, exponent 8, mantissa-high 15, mantissa-low 8 of 32; uint8 code bits equal;
graph id_high/id_low equal), so a random flip's tag distribution is respected.

Inputs (all already on disk; nothing is re-injected):
  --phase1-map  artifacts/phase1/vuln_map.csv            (FLAT/IVF_FLAT/IVF_SQ8/HNSW/HNSW_SQ8)
  --pq-map      <out>/vuln_map_pq.csv                    (Tier 1; optional)
  --graph-map   <out>/graph_edges_characterization.csv  (Tier 2; optional)
  region sizes  artifacts/phase1/regions_aug/*.regions.json (authoritative region_bits)

Outputs (under <out>/rollup/):
  curveB_collapse_prob.csv   PRIMARY — P(collapse) by index over the rate grid
  curveA_expected_dR.csv     APPENDIX — expected ΔR@10 by index (incl. IVF_SQ8 vs HNSW_SQ8)
  region_terms.csv           per-(index,region) E_dR / cat_frac / cat_bits attribution
  validate_multiflip.csv     only with --validate-multiflip (small empirical non-additivity check)

Usage:
  python phase2_rollup.py                       # uses full maps under artifacts/phase2
  python phase2_rollup.py --smoke               # tolerant of missing Tier1/Tier2 maps
  python phase2_rollup.py --validate-multiflip --multiflip-trials 5
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

from qp import config, buckets
from phase1_region_accounting import load_augmented_regions


# bit-share of each fp32 within-element tag (sums to 32).
FP32_W = {"sign": 1.0, "exponent": 8.0, "mantissa-high": 15.0, "mantissa-low": 8.0}


def log(msg):
    print(msg, flush=True)


def read_csv(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(row, *keys):
    """First present, non-empty float among `keys`, else None."""
    for k in keys:
        v = row.get(k, "")
        if v not in ("", None):
            try:
                return float(v)
            except ValueError:
                pass
    return None


def dtype_for(index, kind):
    return buckets.element_dtype_of(index, {"kind": kind})


def tag_weights(dtype, tags):
    """Normalized bit-share weight per tag present for a region's element dtype."""
    if dtype == "float32":
        w = {t: FP32_W[t] for t in tags if t in FP32_W}
    else:                                  # uint8 code bits / int32 id_high|id_low — equal
        w = {t: 1.0 for t in tags}
    s = sum(w.values()) or 1.0
    return {t: v / s for t, v in w.items()}


def region_bits_lookup(art_root):
    """Authoritative region_bits per (index, region) from regions_aug."""
    out = {}
    for spec in config.INDEX_SPECS:
        name = spec["name"]
        try:
            rmap = load_augmented_regions(
                name, os.path.join(art_root, "phase1", "regions_aug"))
        except FileNotFoundError:
            continue
        for r in rmap["regions"]:
            out[(name, r["name"])] = buckets.region_bits(r)
    return out


# --- collapse per-tag map rows into per-region E_dR + catastrophic fraction ----------
def regions_from_tag_map(rows, rbits):
    """For phase1 / pq maps: group (index,region) and bit-weight the tags."""
    groups = {}
    for r in rows:
        groups.setdefault((r["index"], r["region"], r["kind"]), []).append(r)
    out = []
    for (index, region, kind), recs in groups.items():
        dtype = dtype_for(index, kind)
        tags = [rc["bit_position_tag"] for rc in recs]
        w = tag_weights(dtype, tags)
        e_dr = cat = 0.0
        for rc in recs:
            wt = w.get(rc["bit_position_tag"], 0.0)
            e_dr += wt * (_f(rc, "mean_dRecall@10", "mean_dR@10") or 0.0)
            cat += wt * ((_f(rc, "pct_catastrophic") or 0.0) / 100.0)
        rb = rbits.get((index, region))
        if rb is None:
            rb = _f(recs[0], "region_bits") or 0.0
            if not rb:
                log(f"[phase2-T3] [warn] no region_bits for {index}/{region}; "
                    f"treated as 0 (region dropped from curves)")
        out.append({"index": index, "region": region, "kind": kind,
                    "region_bits": rb, "E_dR": e_dr, "cat_frac": cat, "series": "main",
                    "source": "tag_map", "baseline_mode": recs[0].get("baseline_mode", "aligned")})
    return out


def regions_from_graph_map(rows, rbits):
    """For the Tier 2 characterization: use the ALL-tag row. Crashes are DETECTABLE, so the
    silent-collapse contribution to Curve B is pct_silent_harmful (not pct_crash)."""
    out = []
    for r in rows:
        if r.get("bit_position_tag") != "ALL":
            continue
        index, region = r["index"], r["region"]
        rb = rbits.get((index, region))
        if rb is None:
            rb = _f(r, "region_bits") or 0.0
        # series="graph_edges": HNSW and HNSW_SQ8 share an identical graph_edges table, so it
        # is a CONTROLLED variable, not an encoding effect. It is reported as its own curve and
        # kept OUT of the primary per-index collapse curve (see synthesize) so its huge bit
        # count cannot swamp the sq_scale/pq_codebook quantization signal or make fp32 indexes
        # appear to collapse. Crashes (detectable) are already excluded: cat_frac uses
        # silent-harmful only.
        out.append({"index": index, "region": region, "kind": "graph_edges",
                    "region_bits": rb, "E_dR": _f(r, "mean_dR@10") or 0.0,
                    "cat_frac": (_f(r, "pct_silent_harmful") or 0.0) / 100.0, "series": "graph_edges",
                    "source": "graph_map", "baseline_mode": "aligned"})
    return out


# --- curve synthesis ----------------------------------------------------------
def _curves_for(terms_by_index, rate_grid):
    """(curveA, curveB) rows for the given per-index term lists.

    Curve A (expected ΔRecall) is linear superposition — only valid while region_bits·r ≪ 1.
    Past that the sum is unphysical, so we drop non-finite contributions and clamp the total
    to the physical recall-drop range [-1, 1] (recall ∈ [0,1] against a fixed clean baseline).
    """
    curveA, curveB = [], []
    for index in sorted(terms_by_index):
        terms = terms_by_index[index]
        bmode = terms[0]["baseline_mode"]
        for r in rate_grid:
            exp_dr = 0.0
            for t in terms:
                term = t["region_bits"] * r * t["E_dR"]
                if math.isfinite(term):
                    exp_dr += term
            exp_dr = max(-1.0, min(1.0, exp_dr))
            cat_bits_total = sum(t["region_bits"] * t["cat_frac"] for t in terms)
            p_collapse = (1.0 - math.exp(-cat_bits_total * r)
                          if math.isfinite(cat_bits_total) else 1.0)
            curveA.append({"index": index, "baseline_mode": bmode, "r": r,
                           "expected_dRecall@10": exp_dr})
            curveB.append({"index": index, "baseline_mode": bmode, "r": r,
                           "cat_bits_total": cat_bits_total, "P_collapse": p_collapse})
    return curveA, curveB


def synthesize(region_terms, rate_grid):
    """Build the PRIMARY per-index curves (quantization contrast) and a SEPARATE graph_edges
    series. graph_edges is excluded from the primary curves because it is shared identically by
    HNSW/HNSW_SQ8 (a controlled variable) and its ~2e9-bit count would otherwise dominate
    Curve B and make fp32 indexes appear to collapse — contradicting the study's main axis.
    Returns (curveA, curveB, curveA_graph, curveB_graph)."""
    main_by, graph_by = {}, {}
    for t in region_terms:
        (graph_by if t.get("series") == "graph_edges" else main_by) \
            .setdefault(t["index"], []).append(t)
    curveA, curveB = _curves_for(main_by, rate_grid)
    curveA_graph, curveB_graph = _curves_for(graph_by, rate_grid)
    return curveA, curveB, curveA_graph, curveB_graph


def write_csv(path, rows):
    if not rows:
        return
    # Replace any non-finite float (inf/nan) with "" so the CSVs never carry an unparseable
    # token and a downstream JSON re-serialize can't emit invalid Infinity/NaN.
    safe = [{k: ("" if isinstance(v, float) and not math.isfinite(v) else v)
             for k, v in row.items()} for row in rows]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(safe[0].keys()))
        w.writeheader()
        w.writerows(safe)


# --- optional multi-flip non-additivity spot check ----------------------------
def validate_multiflip(art_root, trials, ks, queries, seed):
    """Flip k bits at once in sq_scale / pq_codebook and compare combined ΔR to the sum of
    the same k single-bit ΔRs. Empirically checks the linear-superposition assumption Curve A
    leans on. In-process (these regions deserialize cleanly)."""
    import faiss
    from qp import data, metrics
    from qp import indexes as ix
    from qp.flip import to_buffer, flip_bits, rebuild, safe_search
    from phase1_sensitivity import clean_baseline

    targets = [("IVF_SQ8", "sq_scale"), ("HNSW_SQ8", "sq_scale"),
               ("IVF_PQ_M8", "pq_codebook"), ("IVF_PQ_M16", "pq_codebook")]
    spec_by = {s["name"]: s for s in config.INDEX_SPECS}
    with open(os.path.join(art_root, "baseline.json")) as f:
        knob_by = {r["index"]: r["knob_value"] for r in json.load(f)["indexes"]}
    xq = data.load_query()[:queries].astype("float32")
    gt_ids = np.load(os.path.join(art_root, "gt", "gt_ids.npy"))[:queries]
    gt_dist = np.load(os.path.join(art_root, "gt", "gt_dist.npy"))[:queries]

    def measure(buf, spec, knob):
        idx, exc = rebuild(buf)
        if exc is not None:
            return None
        ix.set_knob(idx, spec, knob)
        _, I, sexc = safe_search(idx, xq, 100)
        if sexc is not None:
            return None
        return metrics.recall_curve(I, gt_ids, config.RECALL_KS)[10]

    rows = []
    for ti, (name, region_name) in enumerate(targets):
        spec, knob = spec_by[name], knob_by[name]
        index = faiss.read_index(os.path.join(art_root, "indexes", f"{name}.faissindex"))
        ix.set_knob(index, spec, knob)
        ref = to_buffer(index); del index
        clean = clean_baseline(ref, spec, knob, xq, gt_ids, gt_dist)["recall"][10]
        rmap = load_augmented_regions(name, os.path.join(art_root, "phase1", "regions_aug"))
        region = next((r for r in rmap["regions"] if r["name"] == region_name), None)
        if region is None:
            continue
        # Stable per-target stream id (targets is fixed-order) — reproducible across processes,
        # unlike builtin hash(name) which is PYTHONHASHSEED-salted.
        rng = np.random.default_rng([seed, ti])
        for k in ks:
            for t in range(trials):
                positions = []
                for _ in range(k):
                    off = int(rng.integers(region["byte_len"]))
                    positions.append((region["byte_start"] + off, int(rng.integers(8))))
                # combined
                flip_bits(ref, positions)
                combined = clean - (measure(ref, spec, knob) or 0.0)
                flip_bits(ref, positions)
                # singles
                ssum = 0.0
                for p in positions:
                    flip_bits(ref, [p])
                    ssum += clean - (measure(ref, spec, knob) or 0.0)
                    flip_bits(ref, [p])
                rows.append({"index": name, "region": region_name, "k": k, "trial": t,
                             "combined_dR@10": combined, "sum_singles_dR@10": ssum,
                             "ratio": (combined / ssum) if ssum else None})
        log(f"        multiflip {name}/{region_name}: {trials} trials x k={ks} done")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Phase 2 Tier 3: faults/MB rollup (analysis only).")
    ap.add_argument("--phase1-map", default=None)
    ap.add_argument("--pq-map", default=None)
    ap.add_argument("--graph-map", default=None)
    ap.add_argument("--rates", default=None,
                    help="comma-separated per-bit rates (default config.PHASE2_RATE_GRID)")
    ap.add_argument("--validate-multiflip", action="store_true")
    ap.add_argument("--multiflip-trials", type=int, default=5)
    ap.add_argument("--multiflip-ks", default="2,3,4")
    ap.add_argument("--multiflip-queries", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    art_root = os.path.join(config.ROOT, "artifacts")
    out = os.path.abspath(args.out or (os.path.join(config.ROOT, "artifacts_smoke", "phase2")
                                       if args.smoke else os.path.join(art_root, "phase2")))
    rollup_dir = os.path.join(out, "rollup")
    os.makedirs(rollup_dir, exist_ok=True)

    phase1_map = args.phase1_map or os.path.join(art_root, "phase1", "vuln_map.csv")
    pq_map = args.pq_map or os.path.join(out, "vuln_map_pq.csv")
    graph_map = args.graph_map or os.path.join(out, "graph_edges_characterization.csv")
    rate_grid = ([float(x) for x in args.rates.split(",")] if args.rates
                 else list(config.PHASE2_RATE_GRID))

    rbits = region_bits_lookup(art_root)
    region_terms = []
    p1 = read_csv(phase1_map)
    if p1:
        region_terms += regions_from_tag_map(p1, rbits)
        log(f"[phase2-T3] phase1 map: {len(p1)} tag rows -> {len(region_terms)} regions")
    else:
        log(f"[phase2-T3] [warn] phase1 map missing: {phase1_map}")
    pq = read_csv(pq_map)
    if pq:
        add = regions_from_tag_map(pq, rbits)
        region_terms += add
        log(f"[phase2-T3] pq map: {len(pq)} tag rows -> +{len(add)} regions")
    else:
        log(f"[phase2-T3] [warn] pq map missing (Tier 1 not run yet?): {pq_map}")
    gm = read_csv(graph_map)
    if gm:
        add = regions_from_graph_map(gm, rbits)
        region_terms += add
        log(f"[phase2-T3] graph map: +{len(add)} graph_edges regions")
    else:
        log(f"[phase2-T3] [warn] graph map missing (Tier 2 not run yet?): {graph_map}")

    curveA, curveB, curveA_graph, curveB_graph = synthesize(region_terms, rate_grid)
    write_csv(os.path.join(rollup_dir, "curveA_expected_dR.csv"), curveA)
    write_csv(os.path.join(rollup_dir, "curveB_collapse_prob.csv"), curveB)
    # graph_edges reported as its own series, kept out of the primary quantization-contrast curves.
    write_csv(os.path.join(rollup_dir, "curveA_graph_edges.csv"), curveA_graph)
    write_csv(os.path.join(rollup_dir, "curveB_graph_edges.csv"), curveB_graph)
    write_csv(os.path.join(rollup_dir, "region_terms.csv"),
              [{k: t[k] for k in ("index", "region", "kind", "region_bits",
                                  "E_dR", "cat_frac", "series", "source", "baseline_mode")}
               for t in sorted(region_terms, key=lambda x: (x["index"], x["region"]))])
    log("[phase2-T3] wrote rollup/{curveA_expected_dR,curveB_collapse_prob,"
        "curveA_graph_edges,curveB_graph_edges,region_terms}.csv")

    # headline preview: P_collapse at the top of the rate grid, by index.
    rmax = max(rate_grid)
    log(f"[phase2-T3] Curve B preview at r={rmax:g} (P collapse):")
    for row in curveB:
        if row["r"] == rmax:
            log(f"        {row['index']:<12} cat_bits={row['cat_bits_total']:.1f}  "
                f"P_collapse={row['P_collapse']:.3e}  [{row['baseline_mode']}]")

    if args.validate_multiflip:
        log("[phase2-T3] multi-flip non-additivity spot check ...")
        ks = [int(x) for x in args.multiflip_ks.split(",")]
        mf = validate_multiflip(art_root, args.multiflip_trials, ks,
                                args.multiflip_queries, args.seed)
        write_csv(os.path.join(rollup_dir, "validate_multiflip.csv"), mf)
        log(f"[phase2-T3] wrote rollup/validate_multiflip.csv ({len(mf)} rows)")

    if args.smoke:
        assert curveA and curveB, "rollup produced no curves"
        log("\nPHASE2-T3 SMOKE OK")
        return 0
    log("\nPHASE2-T3 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
