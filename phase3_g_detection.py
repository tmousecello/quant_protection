"""Phase 3 Stage 3 — task #4: G offline analysis — detection outputs the decision variable.

Zero new computation: reads the existing expb records (4 patterns x {light, sev384}) and
compares, per (pattern, f, severity, recovery):

  primary   observed_load  = stats.load.elements_crc_fail / stats.load.elements_checked
            vs the injected truth (fraction_actual). The load-time CRC scan checks every
            element, so this ratio equals the truth EXACTLY BY CONSTRUCTION (the in-run
            assertion _check_crc_fail_count already enforced crc_fail == n_elements).
            That exactness IS the §3.3 claim — the detector's output is directly the
            decision variable f̂ the scrub/EB policy consumes; report it honestly as
            such, not as a noisy estimate.

  secondary observed_access = stats.totals.corrupt_hits / stats.totals.consults — the
            access-weighted view during search (what fraction of consulted vectors was
            corrupt). Informational: query traffic is not uniform over elements.

`recovery == none` rows are excluded (elements_checked == 0 by design — no detection ran)
and counted in the summary.

Outputs (default artifacts/phase3/g_detection/): detection_table.csv / .md, a diagonal
scatter figure, g_detection_summary.json.
"""

import argparse
import csv
import glob
import json
import os

from qp import config

DETECTING_MODES = ("drop", "fallback_eb")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def extract_points(rows, severity):
    """Detection points from expb rows; returns (points, n_excluded_none).

    A point carries the truth (fraction_actual; nominal fraction for reference) and the
    two observed ratios. Rows without a load scan (recovery=none) are excluded.
    """
    points, excluded = [], 0
    for row in rows:
        stats = row.get("stats") or {}
        load = stats.get("load") or {}
        checked = int(load.get("elements_checked", 0))
        if row.get("recovery") not in DETECTING_MODES or checked <= 0:
            excluded += 1
            continue
        totals = stats.get("totals") or {}
        consults = int(totals.get("consults", 0))
        points.append({
            "pattern": row["pattern"],
            "severity": severity,
            "recovery": row["recovery"],
            "fraction_nominal": float(row["fraction"]),
            "truth_fraction_actual": float(row["fraction_actual"]),
            "elements_checked": checked,
            "elements_crc_fail": int(load.get("elements_crc_fail", 0)),
            "observed_load": int(load.get("elements_crc_fail", 0)) / checked,
            "observed_access": (int(totals.get("corrupt_hits", 0)) / consults
                                if consults > 0 else None),
        })
    return points, excluded


def deviation_stats(points):
    """|observed_load - truth| aggregate (expected ~0 by construction)."""
    devs = [abs(p["observed_load"] - p["truth_fraction_actual"]) for p in points]
    dev_nom = [abs(p["observed_load"] - p["fraction_nominal"]) for p in points]
    if not devs:
        return {"n_points": 0}
    return {
        "n_points": len(devs),
        "max_abs_dev_vs_truth": max(devs),
        "mean_abs_dev_vs_truth": sum(devs) / len(devs),
        "max_abs_dev_vs_nominal": max(dev_nom),
        "identical_to_truth": all(d == 0.0 for d in devs),
    }


# ---------------------------------------------------------------------------
# IO + outputs
# ---------------------------------------------------------------------------

def _load_severity_group(records_dir, suffix, severity):
    paths = sorted(glob.glob(os.path.join(records_dir, f"expb_*_ex_code{suffix}.records.jsonl")))
    rows = []
    for p in paths:
        with open(p) as fh:
            rows.extend(json.loads(l) for l in fh if l.strip())
    return paths, rows


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    all_points, excluded_none, files = [], 0, {}
    for suffix, severity in (("", "light"), ("_sev384", "sev384")):
        paths, rows = _load_severity_group(args.records_dir, suffix, severity)
        pts, excl = extract_points(rows, severity)
        all_points.extend(pts)
        excluded_none += excl
        files[severity] = paths
    if not all_points:
        raise RuntimeError(
            f"REPORT-AND-STOP: no detection points extractable from {args.records_dir} — "
            f"fields missing or only recovery=none rows; per spec, downgrade to a "
            f"one-line limitation instead of re-running.")

    all_points.sort(key=lambda p: (p["severity"], p["pattern"],
                                   p["truth_fraction_actual"], p["recovery"]))
    cols = ["severity", "pattern", "recovery", "fraction_nominal", "truth_fraction_actual",
            "elements_checked", "elements_crc_fail", "observed_load", "observed_access"]
    csv_path = os.path.join(args.out_dir, "detection_table.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_points)

    md_path = os.path.join(args.out_dir, "detection_table.md")
    with open(md_path, "w") as fh:
        fh.write("# G — on-access detection vs injected truth\n\n"
                 "observed_load = elements_crc_fail/elements_checked (load-time CRC scan; "
                 "equals truth by construction — this exactness is the §3.3 claim: the "
                 "detector directly emits the decision variable). observed_access = "
                 "corrupt_hits/consults (access-weighted, informational).\n\n")
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("|" + "---|" * len(cols) + "\n")
        for p in all_points:
            fh.write("| " + " | ".join(
                (f"{p[c]:.5f}" if isinstance(p[c], float) else str(p[c])) for c in cols)
                + " |\n")

    fig_path = os.path.join(args.out_dir, "fig_detection_diagonal.png")
    _draw(all_points, fig_path)

    summary = {
        "deviation": deviation_stats(all_points),
        "n_points": len(all_points),
        "excluded_rows_recovery_none_or_unchecked": excluded_none,
        "severities_found": {sev: len(ps) for sev, ps in files.items()},
        "source_files": files,
        "outputs": {"csv": csv_path, "md": md_path, "figure": fig_path},
        "note": ("load-scan ratio equals injected truth by construction (every element "
                 "CRC-checked at load; in-run assertion enforced equality) — evidence for "
                 "§3.3 'detection directly emits the decision variable'. The access-weighted "
                 "ratio differs from f because query traffic is non-uniform."),
    }
    sum_path = os.path.join(args.out_dir, "g_detection_summary.json")
    with open(sum_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[g-detect] {len(all_points)} points "
          f"(excluded {excluded_none} none/unchecked rows) -> {args.out_dir}")
    d = summary["deviation"]
    print(f"  observed_load vs truth: max|dev|={d['max_abs_dev_vs_truth']:.2e} "
          f"identical={d['identical_to_truth']}")
    return summary


def _draw(points, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    marker = {"light": "o", "sev384": "s"}
    color = {"drop": "#d62728", "fallback_eb": "#1f77b4"}
    lim = max(p["truth_fraction_actual"] for p in points) * 1.1
    for ax, ykey, title in ((ax1, "observed_load", "load-scan (decision variable)"),
                            (ax2, "observed_access", "access-weighted (informational)")):
        ax.plot([0, lim], [0, lim], color="gray", lw=0.8, ls=":")
        for p in points:
            y = p[ykey]
            if y is None:
                continue
            ax.scatter(p["truth_fraction_actual"], y, s=22, alpha=0.7,
                       marker=marker[p["severity"]], color=color[p["recovery"]])
        ax.set_xlabel("injected truth f (fraction_actual)")
        ax.set_ylabel(ykey)
        ax.set_title(title, fontsize=9)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=m)
               for m, c in color.items()]
    handles += [plt.Line2D([], [], color="k", marker=mk, ls="", label=sv)
                for sv, mk in marker.items()]
    ax1.legend(handles=handles, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="G: detection-vs-truth offline analysis")
    ap.add_argument("--records-dir",
                    default=os.path.join(config.ROOT, "artifacts", "phase3", "expb"))
    ap.add_argument("--out-dir",
                    default=os.path.join(config.ROOT, "artifacts", "phase3", "g_detection"))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
