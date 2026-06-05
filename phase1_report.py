#!/usr/bin/env python3
"""Phase 1 — Task C: tables and charts from the single-bit sensitivity sweep.

Reads the raw per-flip shards (raw/<NAME>.records.jsonl) and the aggregated vuln_map and
emits:
  C1  sensitivity_table.{csv,md}   per index x region: mean / p99 / max ΔRecall@10, %catastrophic
  C2  charts/fp32_profile.png      ΔRecall@10 vs fp32 bit tag (sign/exp catastrophic, mantissa-low benign)
      charts/sq8_profile.png       ΔRecall@10 vs SQ8 bit index (MSB->LSB gradient)
      charts/pair_ivf.png          IVF_FLAT (fp32 codes) vs IVF_SQ8 (sq8 codes), shared y
      charts/pair_hnsw.png         HNSW (fp32 vectors) vs HNSW_SQ8 (sq8 codes), shared y
  C3  charts/failure_modes.png + failure_modes.csv   per index/region failure-mode mix

Re-runnable without re-doing the (expensive) sweep.

Usage:
  python phase1_report.py [--out artifacts/phase1]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qp import buckets, metrics, config


def log(msg):
    print(msg, flush=True)


def load_raw(out):
    rows = []
    for path in sorted(glob.glob(os.path.join(out, "raw", "*.records.jsonl"))):
        with open(path) as f:
            rows.extend(json.loads(line) for line in f)
    if not rows:
        raise SystemExit(f"no raw records under {out}/raw — run phase1_sensitivity.py first")
    return pd.DataFrame(rows)


# --- C1 -----------------------------------------------------------------------
def c1_table(df, out):
    g = df.groupby(["index", "region", "kind"], sort=False)
    rows = []
    for (index, region, kind), sub in g:
        d10 = sub["dRecall@10"].dropna().to_numpy(dtype=float)
        n = d10.size
        rows.append({
            "index": index, "region": region, "kind": kind,
            "element_dtype": sub["element_dtype"].iloc[0],
            "n": len(sub),
            "mean_dRecall@10": d10.mean() if n else np.nan,
            "p99_dRecall@10": np.percentile(d10, 99) if n else np.nan,
            "max_dRecall@10": d10.max() if n else np.nan,
            "pct_catastrophic": (d10 > buckets.CATASTROPHIC_ABS).mean() * 100 if n else np.nan,
            "pct_benign": (np.abs(d10) <= buckets.BENIGN_ABS).mean() * 100 if n else np.nan,
            "n_crash": int((sub["failure_mode"] == metrics.CRASH).sum()),
            "n_nan_inf": int((sub["failure_mode"] == metrics.NAN_INF).sum()),
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(out, "sensitivity_table.csv"), index=False)

    lines = ["# Phase 1 — Single-Bit Sensitivity (Task C1)\n",
             "ΔRecall@10 = clean − faulted (per index's own clean baseline). "
             f"Catastrophic = ΔRecall@10 > {buckets.CATASTROPHIC_ABS}; "
             f"benign = |ΔRecall@10| ≤ {buckets.BENIGN_ABS}.\n",
             "| index | region | dtype | n | mean | p99 | max | %catastrophic | %benign |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['index']} | {r['region']} | {r['element_dtype']} | {r['n']} | "
            f"{r['mean_dRecall@10']:.5f} | {r['p99_dRecall@10']:.4f} | "
            f"{r['max_dRecall@10']:.4f} | {r['pct_catastrophic']:.1f}% | "
            f"{r['pct_benign']:.1f}% |")
    with open(os.path.join(out, "sensitivity_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return tbl


# --- C2 -----------------------------------------------------------------------
def _profile(df, dtype, tags):
    """mean ΔRecall@10 per (index,region) over the ordered tag list for one element dtype."""
    sub = df[df["element_dtype"] == dtype]
    series = {}
    for (index, region), g in sub.groupby(["index", "region"], sort=False):
        m = g.groupby("bit_position_tag")["dRecall@10"].mean()
        series[f"{index}/{region}"] = [m.get(t, np.nan) for t in tags]
    return series


def _plot_profile(ax, series, tags, title):
    x = np.arange(len(tags))
    for label, ys in series.items():
        ax.plot(x, ys, marker="o", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=30, ha="right")
    ax.set_ylabel("mean ΔRecall@10")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=7)


def c2_charts(df, out):
    cdir = os.path.join(out, "charts")
    os.makedirs(cdir, exist_ok=True)

    fp32 = _profile(df, "float32", buckets.FP32_TAGS)
    if fp32:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _plot_profile(ax, fp32, buckets.FP32_TAGS,
                      "fp32 bit-position profile (sign/exp catastrophic, mantissa-low benign)")
        fig.tight_layout(); fig.savefig(os.path.join(cdir, "fp32_profile.png"), dpi=130)
        plt.close(fig)

    sq8 = _profile(df, "uint8", buckets.SQ8_TAGS)
    if sq8:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _plot_profile(ax, sq8, buckets.SQ8_TAGS, "SQ8 bit-position profile (MSB→LSB gradient)")
        fig.tight_layout(); fig.savefig(os.path.join(cdir, "sq8_profile.png"), dpi=130)
        plt.close(fig)

    _pair_chart(df, "IVF_FLAT", "IVF_SQ8", "codes",
                os.path.join(cdir, "pair_ivf.png"),
                "IVF_FLAT vs IVF_SQ8 — same IVF structure, fp32 vs SQ8 codes")
    _pair_chart(df, "HNSW", "HNSW_SQ8", None,
                os.path.join(cdir, "pair_hnsw.png"),
                "HNSW vs HNSW_SQ8 — same graph family, fp32 vs SQ8 storage")


def _pair_chart(df, fp32_index, sq8_index, region_hint, path, suptitle):
    """Two shared-y panels: fp32 index's fp32 region | sq8 index's sq8 codes."""
    left = df[(df["index"] == fp32_index) & (df["element_dtype"] == "float32")]
    right = df[(df["index"] == sq8_index) & (df["element_dtype"] == "uint8")]
    if left.empty and right.empty:
        return
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    ls = _profile(left, "float32", buckets.FP32_TAGS)
    _plot_profile(axl, ls, buckets.FP32_TAGS, f"{fp32_index} (fp32)")
    rs = _profile(right, "uint8", buckets.SQ8_TAGS)
    _plot_profile(axr, rs, buckets.SQ8_TAGS, f"{sq8_index} (SQ8)")
    axr.set_ylabel("")
    fig.suptitle(suptitle)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --- C3 -----------------------------------------------------------------------
def c3_failure_modes(df, out):
    modes = [metrics.CLEAN, metrics.SILENT_WRONG, metrics.NAN_INF, metrics.CRASH]
    df = df.copy()
    df["ir"] = df["index"] + "/" + df["region"]
    ct = (df.groupby(["ir", "failure_mode"]).size().unstack(fill_value=0)
            .reindex(columns=modes, fill_value=0))
    ct.to_csv(os.path.join(out, "failure_modes.csv"))

    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(ct) + 1)))
    bottom = np.zeros(len(ct))
    y = np.arange(len(ct))
    colors = {metrics.CLEAN: "#4c9f70", metrics.SILENT_WRONG: "#e0a458",
              metrics.NAN_INF: "#c44e52", metrics.CRASH: "#8172b3"}
    for m in modes:
        vals = ct[m].to_numpy(dtype=float)
        ax.barh(y, vals, left=bottom, label=m, color=colors[m])
        bottom += vals
    ax.set_yticks(y); ax.set_yticklabels(ct.index, fontsize=8)
    ax.set_xlabel("flips"); ax.set_title("Failure-mode distribution per index/region")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "charts", "failure_modes.png"), dpi=130)
    plt.close(fig)
    return ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(config.ROOT, "artifacts", "phase1"))
    args = ap.parse_args()
    out = os.path.abspath(args.out)

    log(f"[phase1-C] reading raw shards from {out}/raw")
    df = load_raw(out)
    log(f"        {len(df)} flips across {df['index'].nunique()} indexes")

    c1_table(df, out)
    log("        C1 sensitivity_table.{csv,md} written")
    c2_charts(df, out)
    log("        C2 charts/*.png written")
    c3_failure_modes(df, out)
    log("        C3 failure_modes.{csv,png} written")

    log("\nPHASE1-C OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
