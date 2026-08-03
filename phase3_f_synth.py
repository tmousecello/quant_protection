"""Phase 3 Stage 3 — task #3: F synthesis (frontier / analysis tables) + validation points.

Offline synthesis (no new measurement) combining:
  1. the cliff timelines (run 2 no-protection / run 3 two-layer / run 4 fuse / stage-3
     self-scrub) -> the 4-line main-figure data, aligned on the tick axis;
  2. the expb tolerance curves (LIGHT records only, 4-pattern average, clean anchor at
     f=0) -> f*(R_min) per policy by linear interpolation in f, and the interval ratio
     ln(1-f*_EB)/ln(1-f*_drop)  — the exact §7.2 method, reproduced programmatically;
  3. the Stage-0 cost harness (phase3_cost.mem_cost) -> protection overhead columns.

Modes (formal invocations live in run_stage3_x86.sh Phase C):
  --synthesize       build main_timeline.csv/png, tolerance_fstar.json, frontier.csv
  --run-cliff-check  validation point (c): deterministic run-2 tick-0 reproduction
                     (6 bits, seed 1234^0 -> recall ~ 0.42309) + a k=8 secondary trial
                     (E3a measured collapse fraction 1.0 at k=8). NOTE: E3a's k=4 point
                     has collapse fraction 0.325 (13/40) — a single 4-bit trial is NOT
                     expected to collapse, so it is cited, not asserted.
  --check            gate the 3 validation points (|Δ| <= 0.01) -> nonzero exit on fail

Interpolation-source discipline: the pinned curve uses expb_<pattern>_ex_code.records.jsonl
ONLY (the untagged light sweep, flips_per_element=4) — never _sev384 / _fstar_check.
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import tempfile

import numpy as np

from qp import config, metrics, provenance
import phase3_cost as cost
from qp.rabitq.registry import get_adapter, adapter_name

PATTERNS = ("uniform_accum", "clustered_accum", "cross_row_accum", "burst_accum")
RMIN_GRID = (0.97, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60)
HEADLINE_RMIN = 0.90

# Global regions the cliff layer replicates, and therefore the ones the rent has to pay for.
# Kept in sync with phase3_e6_shapes.CLIFF_REGIONS_ALLOWED, which is what the sweeps actually
# ran with; changing one without the other would make the reported cost describe a different
# configuration from the reported outcomes.
CLIFF_PROTECTED = ("rotation", "centroids", "header")
CHECK_TOL = 0.01

# Default artifact locations (all overridable via CLI)
E3C_DIR = os.path.join("artifacts", "phase3", "e3c")
E5_DIR = os.path.join("artifacts", "phase3", "e5")
EXPB_DIR = os.path.join("artifacts", "phase3", "expb")
RUN2 = "e3c_uniform_accum_rotation_recall.records.jsonl"
RUN3 = "e5_uniform_accum_rotation.records.jsonl"
RUN4 = "e5_uniform_accum_rotation_replicas_fuse_newlane.records.jsonl"
SCRUB = "e5_uniform_accum_rotation_replicas_scrub_newlane.records.jsonl"

TIMELINE_LABELS = [
    ("unprotected", "run 2: no protection (collapses)"),
    ("per_query_repair", "run 3: two-layer repair (flat)"),
    ("fuse", "fuse, seed 1014: replicas co-accumulate (one-shot fuse)"),
    ("self_scrub", "stage 3: vote-failure-triggered self-scrub (holds)"),
]

# E3a reference (artifacts/phase3/e3a/e3a.json): rotation k=4 collapse fraction 0.325,
# k=8 collapse fraction 1.0. Cited in cliff_check, only k=8 is asserted per-trial.
E3A_K4_COLLAPSE_FRACTION = 0.325


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def load_recall_timeline(path):
    """[(tick, recall)] from a records jsonl with 'tick' and 'recall@10' keys."""
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out.append((int(row["tick"]), row["recall@10"]))
    return out


def align_timelines(series):
    """{label: [(tick, recall)]} -> list of dict rows aligned on tick (missing -> None)."""
    max_tick = max(t for rows in series.values() for t, _ in rows)
    by_label = {lab: dict(rows) for lab, rows in series.items()}
    aligned = []
    for tick in range(max_tick + 1):
        row = {"tick": tick}
        for lab in series:
            row[lab] = by_label[lab].get(tick)
        aligned.append(row)
    return aligned


def _recall_or_stop(row):
    """recall@10 as float, or REPORT-AND-STOP if it is None (a crashed/timed-out expb row).

    Silently skipping a crash row would corrupt the 4-pattern mean; per project policy the
    caller must fix the crash and re-run, not average around it.
    """
    r = row.get("recall@10")
    if r is None:
        raise RuntimeError(
            f"REPORT-AND-STOP: crash/timeout row (recall@10=None) in "
            f"pattern={row.get('pattern')} f={row.get('fraction')} "
            f"recovery={row.get('recovery')} — cannot enter the tolerance curve; "
            f"fix the failing sweep cell and re-run, do not average around it.")
    return float(r)


def tolerance_curves(records_rows, clean_recall):
    """expb rows -> {recovery: [(f, mean recall across patterns)]}, clean anchor prepended.

    Averages recall@10 over the 4 patterns per (fraction, recovery) — the §7.2 method.
    """
    acc = {}
    for row in records_rows:
        key = (row["recovery"], float(row["fraction"]))
        acc.setdefault(key, []).append(_recall_or_stop(row))
    curves = {}
    for (mode, f), vals in acc.items():
        curves.setdefault(mode, []).append((f, sum(vals) / len(vals)))
    for mode in curves:
        curves[mode] = [(0.0, float(clean_recall))] + sorted(curves[mode])
    return curves


def interp_recall(curve, f):
    """Linear interpolation of recall at fraction f on [(f, recall)] (sorted by f)."""
    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    if f < xs[0] or f > xs[-1]:
        raise ValueError(f"f={f} outside curve range [{xs[0]}, {xs[-1]}]")
    return float(np.interp(f, xs, ys))


def solve_fstar(curve, r_min):
    """Largest f with recall(f) >= r_min, by linear interpolation on the LAST downward
    crossing of r_min.

    Scanning from the right (not the first crossing) means a non-monotonic 4-pattern-averaged
    curve that dips below r_min and recovers is not truncated early — f* is the largest f at
    which recall is still >= r_min. Returns None if the curve never drops below r_min (f*
    beyond measured range) or starts below it already. y0 >= r_min > y1 implies y0 > y1, so
    the interpolation denominator is always positive.
    """
    for (f0, y0), (f1, y1) in reversed(list(zip(curve, curve[1:]))):
        if y0 >= r_min > y1:
            return f0 + (y0 - r_min) * (f1 - f0) / (y0 - y1)
    return None


def interval_ratio(f_eb, f_drop):
    """t_EB / t_drop = ln(1-f*_EB) / ln(1-f*_drop) (Bernoulli accumulation, saturating)."""
    return math.log(1.0 - f_eb) / math.log(1.0 - f_drop)


def fstar_table(curves, rmin_grid=RMIN_GRID):
    """Reproduce the §7.2 table: per R_min, f*_drop / f*_EB / interval ratio."""
    table = []
    for r_min in rmin_grid:
        f_drop = solve_fstar(curves["drop"], r_min)
        f_eb = solve_fstar(curves["fallback_eb"], r_min)
        ratio = (interval_ratio(f_eb, f_drop)
                 if (f_eb is not None and f_drop is not None) else None)
        table.append({"r_min": r_min, "fstar_drop": f_drop, "fstar_eb": f_eb,
                      "interval_ratio": ratio})
    return table


def average_measured_at_fraction(rows, fraction, tol=1e-9):
    """{recovery: mean recall across patterns} for rows at the given nominal fraction."""
    acc = {}
    for row in rows:
        if abs(float(row["fraction"]) - fraction) <= tol:
            acc.setdefault(row["recovery"], []).append(_recall_or_stop(row))
    return {mode: sum(v) / len(v) for mode, v in acc.items()}


def run_check_gates(measured, predictions, cliff, tol=CHECK_TOL):
    """The 3 validation gates. Returns (all_ok, [gate dicts])."""
    gates = []

    pred_eb = predictions["pred_eb_at_headline"]
    pred_drop = predictions["pred_drop_at_headline"]
    eb = measured.get("fallback_eb")
    drop = measured.get("drop")

    d_eb = abs(eb - pred_eb) if eb is not None else None
    gates.append({"gate": "a_headline_eb", "measured": eb, "predicted": pred_eb,
                  "delta": d_eb, "ok": bool(eb is not None and d_eb <= tol)})

    d_drop = abs(drop - pred_drop) if drop is not None else None
    ok_b = bool(drop is not None and eb is not None
                and d_drop <= tol and drop < eb and drop < HEADLINE_RMIN)
    gates.append({"gate": "b_drop_control", "measured": drop, "predicted": pred_drop,
                  "delta": d_drop, "requires": "drop < eb and drop < 0.90", "ok": ok_b})

    det = cliff["deterministic_6bit"]
    # --check runs only in the formal pipeline (real adapter, run-2 records present), so the
    # run-2 tick-0 reproduction delta MUST be present and within tol — a None delta means the
    # deterministic reproduction was never verified and the gate must fail, not pass vacuously.
    ok_c = bool(det["collapse"]
                and det.get("delta") is not None and det["delta"] <= tol
                and cliff["k8"]["collapse"])
    reason_c = ("run-2 reference missing — deterministic reproduction not verified"
                if det.get("delta") is None else "6-bit reproduces run-2 tick-0 within tol")
    gates.append({"gate": "c_cliff", "deterministic": det, "k8": cliff["k8"],
                  "requires": reason_c, "ok": ok_c})

    return all(g["ok"] for g in gates), gates


# ---------------------------------------------------------------------------
# --synthesize
# ---------------------------------------------------------------------------

def _light_expb_records(expb_dir):
    """The pinned light sweep only: expb_<pattern>_ex_code.records.jsonl (untagged)."""
    paths = sorted(glob.glob(os.path.join(expb_dir, "expb_*_ex_code.records.jsonl")))
    if len(paths) != len(PATTERNS):
        raise RuntimeError(
            f"REPORT-AND-STOP: expected {len(PATTERNS)} light expb record files in "
            f"{expb_dir}, found {len(paths)}: {paths}")
    rows = []
    for p in paths:
        with open(p) as fh:
            rows.extend(json.loads(l) for l in fh if l.strip())
    return rows


def _read_detection_overhead(expb_dir):
    """Query-path detection cost per recovery policy, from the --lazy-gate acceptance run.

    Returns {policy: {"ns_per_query": x|None, "pct_of_search": x|None, ...}} — None (blank in
    the CSV) when expb_lazy_gates.json is absent or the run did not pass, because an unmeasured
    time cost must read as unmeasured, not as zero. The headline is the load-vs-lazy
    search_wall_ns delta; the timer and analytic figures ride along as cross-checks.
    """
    blank = {"ns_per_query": None, "pct_of_search": None, "crc_bytes_per_query": None,
             "timed_ns_per_check": None, "analytic_ns_per_query": None}
    out = {"fallback_eb": dict(blank), "drop": dict(blank), "source": None}
    path = os.path.join(expb_dir, "expb_lazy_gates.json")
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        gates = json.load(fh)
    if not gates.get("all_pass"):
        # A failed alignment run means the two modes disagreed; its overhead numbers describe
        # a search we have not shown to be the same search. Do not propagate them.
        return out
    out["source"] = path
    out["adapter"] = gates.get("adapter")
    for policy, entry in (gates.get("overhead") or {}).items():
        if policy not in out:
            continue
        nq = entry.get("n_queries")
        out[policy] = {
            "ns_per_query": entry.get("headline_delta_ns_per_query"),
            "pct_of_search": entry.get("headline_pct_of_search"),
            "crc_bytes_per_query": (entry["crc_bytes"] / nq
                                    if entry.get("crc_bytes") and nq else None),
            "timed_ns_per_check": entry.get("timed_ns_per_check"),
            "analytic_ns_per_query": entry.get("analytic_ns_per_query"),
        }
    return out


def synthesize(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. main-figure timeline -------------------------------------------------
    scrub_path = args.scrub_records or os.path.join(config.ROOT, E5_DIR, SCRUB)
    paths = {
        "unprotected": os.path.join(config.ROOT, E3C_DIR, RUN2),
        "per_query_repair": os.path.join(config.ROOT, E5_DIR, RUN3),
        "fuse": os.path.join(config.ROOT, E5_DIR, RUN4),
        "self_scrub": scrub_path,
    }
    for lab, p in paths.items():
        if not os.path.isfile(p):
            raise RuntimeError(
                f"REPORT-AND-STOP: timeline source for '{lab}' missing: {p}"
                + ("\n  -> the self-scrub run has not been produced yet; run "
                   "`bash run_stage3_x86.sh A` first (or pass --scrub-records for smoke)."
                   if lab == "self_scrub" else ""))
    # Lane-consistency guard (audit round-2, finding 1): the fuse and self_scrub lines both inject
    # replicas, so they must share the same replica lane — otherwise the paired "only scrub differs"
    # comparison is a two-variable confound (the original bug: fuse on old XOR lane, scrub on new
    # SeedSequence lane). unprotected/per_query_repair inject no replicas, so they carry no lane.
    lanes = {}
    for lab in ("fuse", "self_scrub"):
        with open(paths[lab]) as fh:
            first = json.loads(fh.readline())
        lane = first.get("replica_lane")
        if lane is None:
            raise RuntimeError(
                f"REPORT-AND-STOP: '{lab}' records carry no replica_lane marker "
                f"({paths[lab]}); regenerate on the SeedSequence lane.")
        lanes[lab] = lane
    if len(set(lanes.values())) != 1:
        raise RuntimeError(
            f"REPORT-AND-STOP: replica-lane mismatch across main-figure lines: {lanes}")
    series = {lab: load_recall_timeline(p) for lab, p in paths.items()}
    aligned = align_timelines(series)
    csv_path = os.path.join(out_dir, "main_timeline.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["tick"] + [l for l, _ in TIMELINE_LABELS])
        writer.writeheader()
        writer.writerows(aligned)
    png_path = os.path.join(out_dir, "fig_main_timeline.png")
    _draw_timeline(aligned, png_path)

    # --- 2. tolerance interpolation (§7.2 reproduced) -----------------------------
    with open(os.path.join(args.expb_dir, "expb_summary.json")) as fh:
        clean_recall = float(json.load(fh)["clean_recall@10"])
    rows = _light_expb_records(args.expb_dir)
    curves = tolerance_curves(rows, clean_recall)
    table = fstar_table(curves)
    headline = next(r for r in table if r["r_min"] == HEADLINE_RMIN)
    if headline["fstar_eb"] is None:
        raise RuntimeError("REPORT-AND-STOP: headline f*_EB not solvable from the curves")
    predictions = {
        "headline_r_min": HEADLINE_RMIN,
        "headline_fraction": round(headline["fstar_eb"], 3),   # == the f the sweep measures
        "pred_eb_at_headline": interp_recall(curves["fallback_eb"],
                                             round(headline["fstar_eb"], 3)),
        "pred_drop_at_headline": interp_recall(curves["drop"],
                                               round(headline["fstar_eb"], 3)),
    }
    tol_path = os.path.join(out_dir, "tolerance_fstar.json")
    with open(tol_path, "w") as fh:
        json.dump({"clean_recall@10": clean_recall,
                   "method": "4-pattern mean of LIGHT expb records, clean anchor (0, clean), "
                             "linear interpolation in f; ratio = ln(1-f*_EB)/ln(1-f*_drop)",
                   "curves": {m: curves[m] for m in sorted(curves)},
                   "fstar_table": table,
                   "predictions": predictions}, fh, indent=2)

    # --- 3. cost columns + frontier ------------------------------------------------
    adapter = get_adapter(args.adapter)
    rmap = adapter.region_map()
    n_elem = int(rmap["header"]["cur_element_count"])
    total_bytes = int(rmap["total_bytes"])
    # The cliff layer's rent. All three are global structures read on every load or every query,
    # so each carries R=3 copies plus one CRC anchor. The centroids dominate the total at 8 KB
    # against the rotation's 64 and the header's 156 — and that 8 KB is num_cluster*padded_dim*4,
    # so it is a property of THIS corpus (SIFT1M uses 16 clusters), not a constant.
    cliff_cost = cost.mem_cost(rmap, {r: {"mult": 3, "checksum_bytes": 4}
                                      for r in CLIFF_PROTECTED})
    # slope detection: per-element CRC manifest (crc_manifest.py format: 64-B header
    # + 16 B/entry), lives beside the index — same cost for drop and EB.
    manifest_bytes = 64 + 16 * n_elem
    protect_total = cliff_cost["total_bytes"] + manifest_bytes
    # Query-path detection overhead — a TIME cost, deliberately kept out of the byte totals
    # above. Measured by phase3_expb_recovery.py --lazy-gate (--crc-mode lazy vs load); absent
    # until that has been run, in which case the columns are blank rather than guessed.
    detect = _read_detection_overhead(args.expb_dir)
    frontier_path = os.path.join(out_dir, "frontier.csv")
    with open(frontier_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["r_min", "fstar_drop", "fstar_eb", "interval_ratio",
                         "cliff_protection_bytes", "slope_manifest_bytes",
                         "total_protection_bytes", "protection_pct_of_index",
                         "detect_ns_per_query_eb", "detect_pct_of_search_eb",
                         "detect_ns_per_query_drop", "detect_pct_of_search_drop"])
        for r in table:
            writer.writerow([r["r_min"], r["fstar_drop"], r["fstar_eb"],
                             r["interval_ratio"], cliff_cost["total_bytes"],
                             manifest_bytes, protect_total,
                             round(100.0 * protect_total / total_bytes, 4),
                             detect["fallback_eb"]["ns_per_query"],
                             detect["fallback_eb"]["pct_of_search"],
                             detect["drop"]["ns_per_query"],
                             detect["drop"]["pct_of_search"]])

    summary = {
        "outputs": {"main_timeline_csv": csv_path, "main_timeline_png": png_path,
                    "tolerance_fstar": tol_path, "frontier_csv": frontier_path},
        "timeline_final_recalls": {lab: series[lab][-1][1] for lab in series},
        "predictions": predictions,
        "cost": {"cliff": cliff_cost, "slope_manifest_bytes": manifest_bytes,
                 "index_total_bytes": total_bytes, "n_elements": n_elem,
                 "query_path_detection": detect},
        "meta": provenance.collect_provenance(
            adapter, adapter_name(adapter), {"recall@10": clean_recall},
            {"rmin_grid": list(RMIN_GRID)}, args, rmap),
    }
    sum_path = os.path.join(out_dir, "f_synth_summary.json")
    with open(sum_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[f-synth] synthesize -> {out_dir}")
    print(f"  headline f*={predictions['headline_fraction']}: "
          f"pred EB={predictions['pred_eb_at_headline']:.4f} "
          f"drop={predictions['pred_drop_at_headline']:.4f}")
    for r in table:
        # solve_fstar returns None when f* lies beyond the measured range — print n/a,
        # the frontier row is still valid data (same convention as the CSV's empty cell).
        fd, fe, ratio = (("n/a" if v is None else f"{v:{spec}}")
                         for v, spec in ((r["fstar_drop"], ".3f"),
                                         (r["fstar_eb"], ".3f"),
                                         (r["interval_ratio"], ".2f")))
        print(f"  R_min={r['r_min']:.2f}  f*_drop={fd} f*_EB={fe}  ratio={ratio}")
    return summary


def _draw_timeline(aligned, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = {"unprotected": dict(color="#d62728", ls="--"),
              "per_query_repair": dict(color="#2ca02c", ls="-"),
              "fuse": dict(color="#ff7f0e", ls="-."),
              "self_scrub": dict(color="#1f77b4", ls="-", lw=2)}
    ticks = [r["tick"] for r in aligned]
    for lab, legend in TIMELINE_LABELS:
        ys = [r[lab] if r[lab] is not None else float("nan") for r in aligned]
        ax.plot(ticks, ys, label=legend, **styles[lab])
    ax.set_xlabel("tick")
    ax.set_ylabel("recall@10")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Rotation corruption timelines: protection variants")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# --run-cliff-check  (validation point c)
# ---------------------------------------------------------------------------

def run_cliff_check(args):
    from phase3_e3c_temporal import TemporalCorruptor, _resolve_region, FULL_CFG

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    adapter = get_adapter(args.adapter)
    aname = adapter_name(adapter)
    clean_buf = adapter.serialize_index()
    rmap = adapter.region_map()
    gt = adapter.load_groundtruth()
    region = _resolve_region(rmap, "rotation")
    bs, bl = region

    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
        tmp = f.name

    def _recall_of(buf):
        adapter.deserialize_index(buf, tmp)
        res = adapter.search_corrupted(tmp, k=config.K, ef=2000,
                                       out_path=tmp + ".ids.ivecs")
        ids = res.get("ids")
        return float(metrics.recall_at_k(ids, gt, config.K)) if ids is not None else None

    try:
        clean_recall = _recall_of(clean_buf.copy())

        # (c1) deterministic reproduction of run-2 tick 0: same corruptor, same cfg
        # (p=0.005), same seed lane (cfg seed 1234 ^ tick 0) -> the same 6 bits.
        buf = clean_buf.copy()
        corruptor = TemporalCorruptor(region, {**FULL_CFG, "seed": config.SEED})
        corruptor.inject_step(buf, "uniform_accum", config.SEED ^ 0)
        bits = corruptor.cumulative_corruption()["bits_flipped"]
        recall_6bit = _recall_of(buf)

        ref, delta = None, None
        run2_path = os.path.join(config.ROOT, E3C_DIR, RUN2)
        if aname == "real" and os.path.isfile(run2_path):
            ref = load_recall_timeline(run2_path)[0][1]
            delta = abs(recall_6bit - ref)

        det = {"bits_injected": int(bits), "recall": recall_6bit,
               "run2_tick0_ref": ref, "delta": delta,
               "collapse": bool(recall_6bit < 0.5 * clean_recall)}

        # (c2) secondary: k=8 seeded distinct rotation bits (E3a collapse fraction 1.0)
        buf8 = clean_buf.copy()
        rng = np.random.default_rng(config.SEED)
        positions = rng.choice(bl * 8, size=8, replace=False)
        for pos in positions:
            buf8[bs + int(pos) // 8] ^= np.uint8(1 << (int(pos) % 8))
        recall_k8 = _recall_of(buf8)
        k8 = {"recall": recall_k8, "collapse": bool(recall_k8 < 0.5 * clean_recall)}
    finally:
        # glob tmp* to also sweep search sidecars (.ids.ivecs + .dist.fvecs / .stats.json).
        for p in glob.glob(tmp + "*"):
            if os.path.exists(p):
                os.unlink(p)

    out = {"adapter": aname, "clean_recall@10": clean_recall,
           "deterministic_6bit": det, "k8": k8,
           "e3a_k4_collapse_fraction_reference": E3A_K4_COLLAPSE_FRACTION,
           "note": ("k=4 is cited only: E3a measured collapse in 13/40 trials (0.325), so a "
                    "single 4-bit trial is not expected to collapse; the deterministic gate "
                    "reproduces run-2 tick-0 instead (user-approved deviation from the spec "
                    "wording)")}
    out_path = os.path.join(out_dir, "cliff_check.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[f-synth] cliff check ({aname}) -> {out_path}")
    print(f"  6-bit: recall={recall_6bit:.5f} ref={ref} delta={delta} "
          f"collapse={det['collapse']}   k8: recall={recall_k8:.5f} "
          f"collapse={k8['collapse']}")
    return out


# ---------------------------------------------------------------------------
# --check  (gate all 3 validation points)
# ---------------------------------------------------------------------------

def check(args):
    tol_path = os.path.join(args.out_dir, "tolerance_fstar.json")
    cliff_path = os.path.join(args.out_dir, "cliff_check.json")
    for p in (tol_path, cliff_path):
        if not os.path.isfile(p):
            raise RuntimeError(f"REPORT-AND-STOP: {p} missing; run --synthesize / "
                               f"--run-cliff-check first")
    with open(tol_path) as fh:
        predictions = json.load(fh)["predictions"]
    with open(cliff_path) as fh:
        cliff = json.load(fh)

    fstar_paths = sorted(glob.glob(
        os.path.join(args.expb_dir, "expb_*_ex_code_fstar_check.records.jsonl")))
    if not fstar_paths:
        raise RuntimeError(
            f"REPORT-AND-STOP: no fstar_check records in {args.expb_dir}; run the expb "
            f"validation sweep (run_stage3_x86.sh C) first")
    rows = []
    for p in fstar_paths:
        with open(p) as fh:
            rows.extend(json.loads(l) for l in fh if l.strip())
    measured = average_measured_at_fraction(rows, predictions["headline_fraction"])
    if not measured:
        found = sorted({round(float(r["fraction"]), 4) for r in rows})
        raise RuntimeError(
            f"REPORT-AND-STOP: no fstar_check rows at f={predictions['headline_fraction']} "
            f"(sweep measured fractions {found}). The validation sweep ran at a different "
            f"fraction than the synthesized headline f*_EB — re-run `run_stage3_x86.sh C`, "
            f"which derives --fractions from the synthesize output.")

    all_ok, gates = run_check_gates(measured, predictions, cliff)
    per_pattern = {}
    for row in rows:
        if abs(float(row["fraction"]) - predictions["headline_fraction"]) <= 1e-9:
            per_pattern.setdefault(row["recovery"], {})[row["pattern"]] = row["recall@10"]

    result = {"all_ok": all_ok, "gates": gates, "measured_avg": measured,
              "per_pattern_informational": per_pattern,
              "n_fstar_files": len(fstar_paths), "tolerance": CHECK_TOL}
    out_path = os.path.join(args.out_dir, "f_synth_check.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    for g in gates:
        print(f"[f-synth] gate {g['gate']}: {'PASS' if g['ok'] else 'FAIL'}  "
              + json.dumps({k: v for k, v in g.items() if k not in ('gate', 'ok')},
                           default=str)[:200])
    if not all_ok:
        print("[f-synth] REPORT-AND-STOP: a validation point deviates beyond "
              f"{CHECK_TOL} (measured EB/drop vs the interpolated predictions, or the cliff "
              "reproduction) — investigate the interpolation between measured fractions; "
              "do NOT hand-adjust artifacts.")
        sys.exit(1)
    print(f"[f-synth] all 3 validation gates PASS -> {out_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="F synthesis: frontier tables + validation")
    ap.add_argument("--synthesize", action="store_true")
    ap.add_argument("--run-cliff-check", dest="run_cliff", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--adapter", default="auto", choices=("stub", "real", "auto"))
    ap.add_argument("--expb-dir", default=os.path.join(config.ROOT, EXPB_DIR))
    ap.add_argument("--scrub-records", default=None,
                    help="override the self-scrub timeline path (smoke: point at "
                         "artifacts_smoke/...)")
    ap.add_argument("--out-dir",
                    default=os.path.join(config.ROOT, "artifacts", "phase3", "f_synth"))
    args = ap.parse_args()

    if not (args.synthesize or args.run_cliff or args.check):
        ap.error("pick at least one of --synthesize/--run-cliff-check/--check")
    if args.synthesize:
        synthesize(args)
    if args.run_cliff:
        run_cliff_check(args)
    if args.check:
        check(args)


if __name__ == "__main__":
    main()
