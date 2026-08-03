"""Phase 3 Stage 3 — task #4: G offline analysis — detection outputs the decision variable.

Zero new computation: reads the existing expb records (4 patterns x {light, sev384}) and
compares, per (pattern, f, severity, recovery):

  primary   observed_load  = stats.load.elements_crc_fail / stats.load.elements_checked
            vs the injected truth (fraction_actual). The load-time CRC scan checks every
            element, so this ratio equals the truth EXACTLY BY CONSTRUCTION: the numerator
            is pinned by the in-run assertion _check_crc_fail_count (crc_fail == n_elements)
            and the denominator by extract_points' own check that elements_checked ==
            cur_element_count. The §3.3 claim is thus about the CONSTRUCTION itself — the
            detector's output is directly the decision variable f̂ the scrub/EB policy
            consumes — not a rediscovery of detector accuracy; report it honestly as such.

  secondary observed_access = stats.totals.corrupt_hits / stats.totals.consults — the
            access-weighted view during search (what fraction of consulted vectors was
            corrupt). Informational: query traffic is not uniform over elements.

`recovery == none` rows are excluded (elements_checked == 0 by design — no detection ran)
and counted in the summary. Rows produced with `--crc-mode lazy` are REJECTED, not excluded:
on-access detection observes only the consulted subset, which is an access-weighted quantity
rather than a noisier estimate of this one, so mixing it in would quietly change what the
diagonal means. (Rewriting G for the access-weighted view is deliberately left as future work.)

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
        # G's whole construction is "the load scan checked every element", so it is only
        # defined for --crc-mode load. Under lazy the scan never runs: load.* is all zeros,
        # which would look like a recovery=none row and get silently EXCLUDED, quietly
        # shrinking the diagonal instead of failing. Records predating the flag have no
        # crc_mode key and are load runs by construction.
        crc_mode = stats.get("crc_mode", "load")
        if crc_mode != "load":
            raise RuntimeError(
                f"REPORT-AND-STOP: row has crc_mode={crc_mode!r} (pattern={row.get('pattern')} "
                f"f={row.get('fraction')} recovery={row.get('recovery')}). G's diagonal holds "
                f"by construction only when the load scan covered every element; on-access "
                f"detection sees just the consulted subset, which is a DIFFERENT (access-"
                f"weighted) quantity, not a noisier version of this one. Re-run the sweep with "
                f"--crc-mode load, or build the access-weighted analysis separately.")
        checked = int(load.get("elements_checked", 0))
        if row.get("recovery") not in DETECTING_MODES or checked <= 0:
            excluded += 1
            continue
        truth = float(row["fraction_actual"])
        # 'observed_load == truth by construction' holds ONLY if the load scan checked every
        # element (elements_checked == cur_element_count). expb asserts the numerator
        # (elements_crc_fail == n_elements) but NOT this denominator; verify it here from
        # n_elements = cur_element_count * fraction_actual. A drifting denominator would make
        # the diagonal claim false, so stop rather than plot a misleading point.
        n_elements = row.get("n_elements")
        if n_elements is not None and truth > 0:
            implied_total = checked * truth   # == n_elements iff checked == cur_element_count
            if abs(implied_total - float(n_elements)) >= 0.5:
                raise RuntimeError(
                    f"REPORT-AND-STOP: elements_checked ({checked}) != cur_element_count for "
                    f"pattern={row['pattern']} f={row['fraction']} recovery={row['recovery']} "
                    f"(implied total {implied_total:.2f} vs n_elements {n_elements}); the load "
                    f"scan did not check every element, so observed_load would NOT equal the "
                    f"injected truth by construction.")
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
                 "observed_load = elements_crc_fail/elements_checked (load-time CRC scan). It "
                 "equals truth by construction — numerator pinned by the in-run assertion "
                 "crc_fail==n_elements, denominator verified here as elements_checked=="
                 "cur_element_count. The §3.3 evidence is that construction (detection "
                 "directly emits the decision variable), not a rediscovery of accuracy. "
                 "observed_access = corrupt_hits/consults (access-weighted, informational)."
                 "\n\n")
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
        "note": ("load-scan ratio equals injected truth by construction: numerator pinned by "
                 "the in-run crc_fail==n_elements assertion, denominator verified as "
                 "elements_checked==cur_element_count (extract_points stops otherwise) — "
                 "evidence for §3.3 'detection directly emits the decision variable', not a "
                 "rediscovery of accuracy. The access-weighted ratio differs from f because "
                 "query traffic is non-uniform."),
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
