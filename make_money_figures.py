#!/usr/bin/env python3
"""Generate three "money figures" from existing Phase 2 corruption data.

No new experiments are run; this only reads the full-run CSVs already on disk
under artifacts/phase2/ and renders PNGs to artifacts/phase2/figures/.

  1. fig_collapse_probability.png  -- P(silent collapse) vs per-bit error rate
  2. fig_expected_dRecall.png      -- expected dRecall@10 vs per-bit error rate (Curve A)
  3. fig_burst_dispersion.png      -- p_collapse & mean_dR vs burst length, per index

Conventions follow phase1_report.py: Agg backend, dpi=130, tight_layout, close.

Usage:
    .venv/bin/python make_money_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive, write-to-file only
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(HERE, "artifacts", "phase2")
ROLLUP = os.path.join(P2, "rollup")
BURST = os.path.join(P2, "burst")
OUT = os.path.join(P2, "figures")

DPI = 130

# One fixed style per index so a given index looks identical across all figures.
# SQ8 (the indexes that actually collapse) get warm/alert colors; fp32 + PQ get
# cool/neutral colors.
INDEX_STYLE = {
    "IVF_SQ8":    dict(color="#c44e52", marker="o", lw=2.0),   # red
    "HNSW_SQ8":   dict(color="#e0a458", marker="s", lw=2.0),   # amber
    "IVF_FLAT":   dict(color="#4c72b0", marker="^", lw=1.6),   # blue
    "FLAT":       dict(color="#4c9f70", marker="v", lw=1.6),   # green
    "HNSW":       dict(color="#8172b3", marker="D", lw=1.6),   # purple
    "IVF_PQ_M8":  dict(color="#937860", marker="P", lw=1.6),   # brown
    "IVF_PQ_M16": dict(color="#64b5cd", marker="X", lw=1.6),   # cyan
}
# Stable plotting order (SQ8 last so they draw on top of the flat-zero lines).
INDEX_ORDER = ["FLAT", "HNSW", "IVF_FLAT", "IVF_PQ_M8", "IVF_PQ_M16",
               "HNSW_SQ8", "IVF_SQ8"]

# Region targeted by each burst sub-study (constant per file; surfaced in the legend).
BURST_INDEXES = ["HNSW_SQ8", "IVF_FLAT", "IVF_PQ_M8"]


def _require(path):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: required data file missing: {path}")
    return path


def _style(idx):
    return INDEX_STYLE.get(idx, dict(color="#999999", marker=".", lw=1.4))


def _ordered(indexes):
    seen = set(indexes)
    return [i for i in INDEX_ORDER if i in seen] + [i for i in indexes if i not in INDEX_ORDER]


def fig_collapse_probability(out_path):
    """Fig 1: P(silent collapse) vs per-bit error rate r (Curve B).

    y uses symlog so genuine zeros render on the baseline while the two SQ8
    curves separate cleanly; x is log.
    """
    df = pd.read_csv(_require(os.path.join(ROLLUP, "curveB_collapse_prob.csv")))
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in _ordered(df["index"].unique()):
        sub = df[df["index"] == idx].sort_values("r")
        collapses = sub["P_collapse"].max() > 0
        st = _style(idx)
        ax.plot(sub["r"], sub["P_collapse"],
                marker=st["marker"] if collapses else None,
                color=st["color"], lw=st["lw"] if collapses else 1.2,
                ls="-" if collapses else "--",
                alpha=1.0 if collapses else 0.55,
                label=idx + ("" if collapses else "  (P=0)"))
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-7)
    ax.set_xlabel("per-bit error rate  r  (faults / bit)")
    ax.set_ylabel("P(silent collapse)")
    ax.set_title("Silent collapse probability vs per-bit error rate\n"
                 "only SQ8 quantized indexes ever collapse")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_expected_dRecall(out_path):
    """Fig 2: expected dRecall@10 vs per-bit error rate r (Curve A)."""
    df = pd.read_csv(_require(os.path.join(ROLLUP, "curveA_expected_dR.csv")))
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in _ordered(df["index"].unique()):
        sub = df[df["index"] == idx].sort_values("r")
        st = _style(idx)
        ax.plot(sub["r"], sub["expected_dRecall@10"],
                marker=st["marker"], color=st["color"], lw=st["lw"], label=idx)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("per-bit error rate  r  (faults / bit)")
    ax.set_ylabel("expected ΔRecall@10")
    ax.set_title("Expected (mean) ΔRecall@10 vs per-bit error rate\n"
                 "averages are benign and clustered — the danger hides in the tail (Fig 1)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _load_burst():
    frames = {}
    for idx in BURST_INDEXES:
        df = pd.read_csv(_require(os.path.join(BURST, f"{idx}_burst.csv")))
        df = df.sort_values("B_bits")
        # region targeted is constant per file; pull it for the legend label
        region = "?"
        try:
            import json
            region = next(iter(json.loads(df["region_hits"].iloc[0]).keys()))
        except Exception:
            pass
        frames[idx] = (df, region)
    return frames


def fig_burst_dispersion(out_path):
    """Fig 3: two-panel burst study.

    Left  : p_silent_collapse vs burst length, crash zone flagged.
    Right : mean_dR@10 vs burst length (crash rows dropped, they have no recall).
    """
    frames = _load_burst()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    for idx in BURST_INDEXES:
        df, region = frames[idx]
        st = _style(idx)
        label = f"{idx} ({region})"
        # crash rows: whole burst crashes (n_crash == n_trials) -> recall undefined
        is_crash = df["n_crash"] >= df["n_trials"]

        # Left panel: collapse probability (crash rows shown as hollow markers)
        axL.plot(df["B_bits"], df["p_silent_collapse"],
                 color=st["color"], lw=st["lw"], marker=st["marker"], label=label)
        if is_crash.any():
            axL.scatter(df.loc[is_crash, "B_bits"], df.loc[is_crash, "p_silent_collapse"],
                        facecolors="none", edgecolors=st["color"], s=110, lw=1.8, zorder=5)

        # Right panel: mean recall drop, crash rows dropped (mean_dR is blank there)
        ok = df[~is_crash].copy()
        ok["mean_dR@10"] = pd.to_numeric(ok["mean_dR@10"], errors="coerce")
        ok = ok.dropna(subset=["mean_dR@10"])
        axR.plot(ok["B_bits"], ok["mean_dR@10"],
                 color=st["color"], lw=st["lw"], marker=st["marker"], label=label)

    # annotate the crash zone on the left panel (HNSW_SQ8 crashes at >=65536 bits)
    crash_b = []
    for idx in BURST_INDEXES:
        df, _ = frames[idx]
        crash_b += df.loc[df["n_crash"] >= df["n_trials"], "B_bits"].tolist()
    if crash_b:
        zmin = min(crash_b)
        axL.axvspan(zmin, max(df["B_bits"]) * 1.5, color="#c44e52", alpha=0.08)
        axL.text(zmin, 0.5, " crash zone\n (hollow = crash)", color="#c44e52",
                 fontsize=8, va="center", ha="left")

    for ax in (axL, axR):
        ax.set_xscale("log")
        ax.set_xlabel("burst length  B  (contiguous bits flipped)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8)
    axL.set_ylabel("p(silent collapse)")
    axL.set_ylim(-0.05, 1.08)
    axL.set_title("Burst collapse risk")
    axR.set_ylabel("mean ΔRecall@10")
    axR.set_title("Burst mean degradation (graceful, non-crash)")
    fig.suptitle("Spatial concentration: clustered flips into SQ8's tiny sq_scale wipe recall;\n"
                 "bursts into IVF_FLAT centroids / IVF_PQ codebook degrade gracefully",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"reading from {P2}")
    print(f"writing to   {OUT}")
    fig_collapse_probability(os.path.join(OUT, "fig_collapse_probability.png"))
    fig_expected_dRecall(os.path.join(OUT, "fig_expected_dRecall.png"))
    fig_burst_dispersion(os.path.join(OUT, "fig_burst_dispersion.png"))
    print("done.")


if __name__ == "__main__":
    main()
