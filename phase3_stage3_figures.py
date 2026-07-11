#!/usr/bin/env python3
"""Stage 3 report figures — reads FROZEN artifacts only (no index loads, no recall
recomputation; the experiments are frozen per stage3_plan.md §執行順序與凍結).

Inputs (all under artifacts/phase3/):
  e5/e5_uniform_accum_rotation_replicas_scrub_newlane.records.jsonl   (task #1, 200 ticks)
  e5_seeds/first_fail_summary.json + *_seed1002full/*_seed1003full records (task #2)
  f_synth/{tolerance_fstar.json, main_timeline.csv, f_synth_check.json}   (task #3)

Outputs: artifacts/phase3/figures/fig_{scrub_timeline,first_fail_distribution,
tolerance_frontier}.png

Conventions follow phase1_report.py / make_money_figures.py: Agg backend,
dpi=130, tight_layout, close. Palette validated for CVD separation and
lightness band (dataviz validator); sub-3:1-contrast hues (orange/yellow)
carry direct labels as relief.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
P3 = ROOT / "artifacts" / "phase3"
OUT_DIR = P3 / "figures"
DPI = 130

# Entity -> color, fixed across all figures (matches f_synth fig_main_timeline:
# blue = self-scrub/EB "protected", orange = fuse, red = unprotected/drop/collapse).
C_PROTECT = "#1f77b4"
C_FUSE = "#ff7f0e"
C_COLLAPSE = "#d62728"
C_NONE = "#eda100"
C_RATIO = "#4a3aa7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"[stage3-figs] wrote {path}")
    return path


def _read_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- figure 1


def fig_scrub_timeline():
    recs = _read_records(P3 / "e5" / "e5_uniform_accum_rotation_replicas_scrub_newlane.records.jsonl")
    ticks = [r["tick"] for r in recs]
    recall = [r["recall@10"] for r in recs]
    # counters are cumulative -> event ticks via consecutive diff
    reload_ticks, prev = [], 0
    for r in recs:
        cur = r["counters"]["cliff_vote_fail_reloads"]
        if cur != prev:
            reload_ticks.append(r["tick"])
            prev = cur
    final = recs[-1]["counters"]

    # fuse trajectory (seed 1014, SeedSequence lane) from the aligned main-timeline table
    fuse_t, fuse_r = [], []
    with open(P3 / "f_synth" / "main_timeline.csv") as f:
        for row in csv.DictReader(f):
            if row["fuse"]:
                fuse_t.append(int(row["tick"]))
                fuse_r.append(float(row["fuse"]))

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.patch.set_facecolor(SURFACE)
    _style_ax(ax)

    ax.plot(fuse_t, fuse_r, color=C_FUSE, ls="-.", lw=2,
            label="fuse, seed 1014 (no replica scrub)")
    ax.plot(ticks, recall, color=C_PROTECT, lw=2.4,
            label="self-scrub (this run, 200 ticks)")

    for i, t in enumerate(reload_ticks):
        ax.axvline(t, color=C_COLLAPSE, lw=1.2, ls=":", alpha=0.85)
        ax.plot([t], [1.03], marker="v", color=C_COLLAPSE, ms=7)
    ax.text(5, 1.06,
            f"vote-fail full reloads (×{len(reload_ticks)}): ticks {', '.join(map(str, reload_ticks))}",
            color=C_COLLAPSE, fontsize=9, ha="left")

    # direct labels (relief for the sub-3:1 orange)
    ax.text(24, 0.70, "first vote failure at tick 21 (= median)\n→ fuse blown, collapses to 0",
            color=C_FUSE, fontsize=9)
    ax.text(120, 0.945, "recall = 0.98376 (= clean), all 200 ticks",
            color=C_PROTECT, fontsize=9, ha="center", va="top")

    ax.text(0.985, 0.45,
            "final counters: irrecoverable = 0\n"
            f"repaired = {final['cliff_repaired']}, reloads = {final['cliff_vote_fail_reloads']}\n"
            f"anchor checks = {final['cliff_anchor_checked']}, mismatch = {final['cliff_anchor_mismatch']}",
            transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color=INK2,
            bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=AXIS))

    ax.set_xlabel("tick")
    ax.set_ylabel("recall@10")
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlim(-3, 202)
    ax.set_title("Task #1 — replica self-scrub keeps recall flat for 200 ticks (uniform_accum × rotation, R=3)")
    ax.legend(loc="center left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, "fig_scrub_timeline.png")


# ---------------------------------------------------------------- figure 2


def fig_first_fail_distribution():
    summ = json.load(open(P3 / "e5_seeds" / "first_fail_summary.json"))
    per_seed = summ["first_fail_tick_per_seed"]
    stats = summ["stats"]
    ana = summ["analytic"]
    observed = sorted(v for v in per_seed.values() if v is not None)
    n_seeds = stats["n_seeds"]
    ticks_per_run = stats["ticks_per_run"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2, ax3):
        _style_ax(ax)

    # -- panel 1: histogram + median/IQR
    ax1.hist(observed, bins=range(0, 121, 10), color=C_PROTECT, edgecolor=SURFACE, lw=2)
    ax1.axvspan(stats["q1"], stats["q3"], color=C_PROTECT, alpha=0.12)
    ax1.axvline(stats["median"], color=INK, lw=1.4, ls="--")
    ax1.text(stats["median"] + 2, ax1.get_ylim()[1] * 0.95,
             f"median = {stats['median']:.0f}\nIQR = [{stats['q1']:.0f}, {stats['q3']:.0f}]",
             fontsize=9, color=INK, va="top")
    ax1.text(0.97, 0.55,
             f"n = {n_seeds} seeds\nobserved = {stats['n_observed']}\n"
             f"censored = {stats['n_censored']} (seed {summ['stats']['censored_seeds'][0]},\n"
             f"no failure in {ticks_per_run} ticks)",
             transform=ax1.transAxes, ha="right", fontsize=8.5, color=INK2,
             bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=AXIS))
    ax1.set_xlabel("first vote-failure tick")
    ax1.set_ylabel("seeds")
    ax1.set_title("First-failure distribution (30 seeds, legacy fuse)")

    # -- panel 2: ECDF vs analytic geometric CDF
    p = ana["p_fail_per_tick"]
    xs = list(range(0, ticks_per_run + 1))
    geo = [1 - (1 - p) ** (t + 1) for t in xs]
    ecdf_y = [sum(1 for v in observed if v <= t) / n_seeds for t in xs]
    ax2.plot(xs, geo, color=C_COLLAPSE, ls="--", lw=2,
             label=f"analytic geometric (p = {p:.4f}/tick)")
    ax2.step(xs, ecdf_y, where="post", color=C_PROTECT, lw=2,
             label="empirical CDF (29/30 observed)")
    ax2.axhline(1.0, color=AXIS, lw=0.8)
    ax2.text(110, 0.84, "censored seed caps\nECDF at 29/30", fontsize=8.5,
             color=INK2, ha="center")
    ax2.set_xlabel("tick")
    ax2.set_ylabel("P(first failure ≤ tick)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title(f"Empirical vs analytic (median {stats['median']:.0f} vs {ana['geometric_median']:.1f})")
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # -- panel 3: extreme-seed full timelines
    def _tl(seed):
        recs = _read_records(P3 / "e5_seeds" / f"e5_uniform_accum_rotation_replicas_seed{seed}full.records.jsonl")
        t = [r["tick"] for r in recs]
        y = [r["recall@10"] for r in recs]
        ff = next((r["tick"] for r in recs if r["counters"]["cliff_irrecoverable"] > 0), None)
        return t, y, ff

    t3, y3, ff3 = _tl(1003)  # earliest first-fail (tick 0)
    t2, y2, ff2 = _tl(1002)  # latest first-fail (tick 112)
    ax3.axhline(0.98376, color=C_PROTECT, ls="--", lw=1.6)
    ax3.text(80, 1.03, "self-scrub reference (flat 0.98376)", color=C_PROTECT,
             fontsize=8.5, ha="center")
    ax3.plot(t2, y2, color=C_FUSE, lw=2, label="seed 1002 (latest fail, tick 112)")
    ax3.plot(t3, y3, color=C_COLLAPSE, lw=2, label="seed 1003 (earliest fail, tick 0)")
    for ff, y, c in ((ff2, y2, C_FUSE), (ff3, y3, C_COLLAPSE)):
        ax3.plot([ff], [y[ff]], marker="o", ms=8, color=c, mec=SURFACE, mew=1.5)
    ax3.set_xlabel("tick")
    ax3.set_ylabel("recall@10")
    ax3.set_ylim(-0.03, 1.10)
    ax3.set_title("After first failure: collapse follows the\nunprotected trajectory (both extremes)")
    ax3.legend(loc="center", fontsize=8.5, framealpha=0.9)

    fig.suptitle("Task #2 — fuse first-failure distribution, 30 seeds (SeedSequence lanes)", y=1.0)
    fig.tight_layout()
    return _save(fig, "fig_first_fail_distribution.png")


# ---------------------------------------------------------------- figure 3


def fig_tolerance_frontier():
    tol = json.load(open(P3 / "f_synth" / "tolerance_fstar.json"))
    chk = json.load(open(P3 / "f_synth" / "f_synth_check.json"))
    curves = tol["curves"]
    table = tol["fstar_table"]
    r90 = next(r for r in table if r["r_min"] == 0.9)
    gates = {g["gate"]: g for g in chk["gates"]}
    f_meas = 0.101  # headline fraction actually swept (fstar_check tag)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2, ax3):
        _style_ax(ax)

    # -- panel 1: tolerance curves + r_min=0.90 operating point + measured stars
    series = [("fallback_eb", C_PROTECT, "-", "EB fallback"),
              ("drop", C_COLLAPSE, "-", "drop"),
              ("none", C_NONE, "-.", "none (no recovery)")]
    for key, color, ls, label in series:
        xs = [p[0] for p in curves[key]]
        ys = [p[1] for p in curves[key]]
        ax1.plot(xs, ys, color=color, ls=ls, lw=2, marker="o", ms=5, label=label)
        ax1.text(xs[-1] + 0.008, ys[-1], label, color=color, fontsize=8.5, va="center")
    ax1.axhline(0.90, color=MUTED, ls="--", lw=1.2)
    ax1.text(0.45, 0.906, "R_min = 0.90", color=INK2, fontsize=8.5)
    for fstar, color in ((r90["fstar_drop"], C_COLLAPSE), (r90["fstar_eb"], C_PROTECT)):
        ax1.plot([fstar], [0.90], marker="|", ms=14, mew=2.5, color=color)
    ax1.text(0.135, 0.925, f"f*_drop = {r90['fstar_drop']:.3f}   f*_EB = {r90['fstar_eb']:.3f}",
             color=INK2, fontsize=8.5, ha="left")
    # measured validation points (gates a & b), stars
    ax1.plot([f_meas], [gates["a_headline_eb"]["measured"]], marker="*", ms=15,
             color=C_PROTECT, mec=INK, mew=0.6, ls="none", zorder=5,
             label="measured @ f = 0.101 (gates a/b)")
    ax1.plot([f_meas], [gates["b_drop_control"]["measured"]], marker="*", ms=15,
             color=C_COLLAPSE, mec=INK, mew=0.6, ls="none", zorder=5)
    ax1.set_xlabel("corrupted-element fraction f (ex_code)")
    ax1.set_ylabel("recall@10")
    ax1.set_xlim(-0.02, 0.68)
    ax1.set_title("Tolerance curves (4-pattern mean) + measured checks")
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    # -- panel 2: f* vs R_min
    rmins = [r["r_min"] for r in table]
    ax2.plot(rmins, [r["fstar_eb"] for r in table], color=C_PROTECT, lw=2, marker="o", ms=6,
             label="f*_EB")
    ax2.plot(rmins, [r["fstar_drop"] for r in table], color=C_COLLAPSE, lw=2, marker="s", ms=6,
             label="f*_drop")
    ax2.invert_xaxis()  # stricter recall floor on the left
    ax2.set_xlabel("recall floor R_min (stricter →left)")
    ax2.set_ylabel("max tolerable corruption f*")
    ax2.set_title("Scrub-interval frontier: f*(R_min)")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # -- panel 3: interval ratio (single series)
    ratios = [r["interval_ratio"] for r in table]
    ax3.plot(rmins, ratios, color=C_RATIO, lw=2, marker="o", ms=6)
    for r, v in zip(rmins, ratios):
        ax3.annotate(f"{v:.2f}", (r, v), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8, color=INK2)
    ax3.invert_xaxis()
    ax3.set_ylim(1.0, 1.30)
    ax3.set_xlabel("recall floor R_min (stricter →left)")
    ax3.set_ylabel("interval ratio  t_EB / t_drop")
    ax3.set_title("EB extends the scrub interval 1.06–1.23×\n(ratio = ln(1−f*_EB) / ln(1−f*_drop))")

    fig.suptitle("Task #3 — F synthesis: tolerance curves, f* frontier, interval ratio", y=1.0)
    fig.tight_layout()
    return _save(fig, "fig_tolerance_frontier.png")


def main():
    fig_scrub_timeline()
    fig_first_fail_distribution()
    fig_tolerance_frontier()


if __name__ == "__main__":
    main()
