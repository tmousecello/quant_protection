"""Roll the centroid-gap sweeps up into the table the paper row is built from.

Reads the E6 shards and the E9 sweep produced by run_centroid_sweeps.sh, and answers the four
questions the plan said would falsify the work:

  1. did protection actually convert the crash and silent-wrong cells?
  2. did any "repaired" trial fail to restore recall (damage escaping the protected window)?
  3. is the rotation column byte-identical to the archived baseline (vectorization regression)?
  4. is the repair attributable to the region that was supposed to do it?
"""
import csv
import glob
import json
import os
import sys

E6 = os.path.join("artifacts", "phase3", "e6")
E9 = os.path.join("artifacts", "phase3", "e9")
BASELINE = ("/home/u01/tmouse/vecdb-recovery-monorepo/results/jonathan-vuln-shapes/"
            "e6_shards")
OUTCOMES = ("crash", "silent_wrong", "detected_wrong", "repaired", "tolerated")


def load_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def hist(rows, arm):
    h = {o: 0 for o in OUTCOMES}
    for r in rows:
        if r["arm"] == arm:
            h[r["outcome"]] += 1
    return h


def fmt(h):
    return "  ".join(f"{o[:2].capitalize()}={h[o]:>2}" for o in OUTCOMES)


def main():
    report = {}

    print("=" * 100)
    print("E6 centroids — outcome by cliff configuration (30 seeds x 2 arms per shape)")
    print("=" * 100)
    print(f"{'shape':<15} {'config':<26} {'arm':<4} {fmt({o: 0 for o in OUTCOMES})}")
    cent = {}
    for shape in ("single_cell", "device_row", "device_column"):
        for tag, label in (("v2cent", "rotation+centroids"),
                           ("v2cent_hdr", "rotation+centroids+header")):
            p = os.path.join(E6, f"e6_results_{shape}__centroids_{tag}.csv")
            if not os.path.isfile(p):
                print(f"{shape:<15} {label:<26} MISSING {p}")
                continue
            rows = load_csv(p)
            for arm in ("off", "on"):
                h = hist(rows, arm)
                cent[(shape, tag, arm)] = h
                print(f"{shape:<15} {label:<26} {arm:<4} {fmt(h)}")
    report["e6_centroids"] = {f"{s}|{t}|{a}": h for (s, t, a), h in cent.items()}

    print()
    print("=" * 100)
    print("E6 rotation — vectorization regression vs the archived baseline shards")
    print("=" * 100)
    rot_ok = True
    rot_rows = {}
    for shape in ("single_cell", "device_row", "device_column"):
        new_p = os.path.join(E6, f"e6_results_{shape}__rotation_v2cent.csv")
        old_p = os.path.join(BASELINE, f"e6_results_{shape}__rotation.csv")
        if not os.path.isfile(new_p):
            print(f"{shape:<15} MISSING {new_p}")
            rot_ok = False
            continue
        new = load_csv(new_p)
        h_new = hist(new, "on")
        line = f"{shape:<15} new  on  {fmt(h_new)}"
        if os.path.isfile(old_p):
            old = load_csv(old_p)
            h_old = hist(old, "on")
            same = h_new == h_old
            rot_ok &= same
            line += f"   | baseline {fmt(h_old)}   {'MATCH' if same else 'DIFFERS'}"
            rot_rows[shape] = {"new": h_new, "baseline": h_old, "match": same}
        else:
            line += "   | baseline shard not found"
            rot_rows[shape] = {"new": h_new, "baseline": None, "match": None}
        print(line)
    report["e6_rotation_regression"] = {"per_shape": rot_rows, "all_match": rot_ok}

    print()
    print("=" * 100)
    print("E9 miscorrection (ef=64) — all strata")
    print("=" * 100)
    e9p = os.path.join(E9, "e9_results_v2cent_hdr.csv")
    if os.path.isfile(e9p):
        rows = load_csv(e9p)
        strata = []
        for s in dict.fromkeys(r["region"] for r in rows):
            sr = [r for r in rows if r["region"] == s]
            print(f"{s:<16} off  {fmt(hist(sr, 'off'))}   |  on  {fmt(hist(sr, 'on'))}")
            strata.append({"region": s, "off": hist(sr, "off"), "on": hist(sr, "on")})
        report["e9"] = strata
    else:
        print(f"MISSING {e9p}")

    print()
    print("=" * 100)
    print("Falsification checks")
    print("=" * 100)

    # Raw records, split by configuration. The centroids-only shards are the ABLATION and are
    # expected to keep crashing on device_row — pooling them with the full configuration would
    # report the ablation's failures as the result's.
    raw_by_cfg = {"rotation+centroids": [], "rotation+centroids+header": []}
    for p in glob.glob(os.path.join(E6, "raw", "*.records.jsonl")) + \
            glob.glob(os.path.join(E9, "raw", "*.records.jsonl")):
        base = os.path.basename(p)
        if "v2cent_hdr" in base:
            key = "rotation+centroids+header"
        elif "v2cent" in base:
            key = "rotation+centroids"
        else:
            continue
        with open(p) as fh:
            raw_by_cfg[key] += [json.loads(x) for x in fh]

    for cfg_name, recs in raw_by_cfg.items():
        on = [r for r in recs if r["arm"] == "on"]
        note = "  (ablation: device_row is expected to keep crashing here)" \
            if cfg_name == "rotation+centroids" else "  (the full configuration)"
        print(f"  {cfg_name}{note}")
        print(f"      protected evals={len(on):>4}  "
              f"crash={sum(1 for r in on if r['outcome'] == 'crash'):>3}  "
              f"silent_wrong={sum(1 for r in on if r['outcome'] == 'silent_wrong'):>3}  "
              f"repaired={sum(1 for r in on if r['outcome'] == 'repaired'):>3}")
    print()

    raw = raw_by_cfg["rotation+centroids+header"]
    repaired = [r for r in raw if r["outcome"] == "repaired"]
    leaky = [r for r in repaired if r["delta_recall"] is not None
             and abs(float(r["delta_recall"])) > 0.005]
    print(f"  repaired trials inspected                : {len(repaired)}")
    print(f"  repaired but recall did NOT return       : {len(leaky)}"
          f"{'   <-- damage escaped the protected window' if leaky else ''}")
    for r in leaky[:5]:
        print(f"      {r['shape']}/{r['region']} seed={r['seed']} "
              f"delta={r['delta_recall']} field_hits={r.get('field_hits')}")

    # (4) attribution: which region did the repairing
    attributed = [r for r in raw if r.get("cliff_by_region")]
    by_region = {}
    for r in attributed:
        for name, n in r["cliff_by_region"].items():
            by_region.setdefault(name, {"trials_acting": 0, "bits": 0})
            if n:
                by_region[name]["trials_acting"] += 1
                by_region[name]["bits"] += n
    print(f"  protected trials with a region breakdown : {len(attributed)}")
    for name, v in sorted(by_region.items()):
        print(f"      {name:<12} acted in {v['trials_acting']:>4} trials, "
              f"{v['bits']:>9,} bits repaired")

    # (1) the headline conversion, in the full configuration only
    crashes = [r for r in raw if r["arm"] == "on" and r["outcome"] == "crash"]
    silent = [r for r in raw if r["arm"] == "on" and r["outcome"] == "silent_wrong"]
    print(f"  protected crashes remaining              : {len(crashes)}")
    print(f"  protected silent-wrong remaining         : {len(silent)}")
    for r in (crashes + silent)[:6]:
        print(f"      {r['shape']}/{r['region']} seed={r['seed']} {r['outcome']} "
              f"err={str(r.get('error'))[:90]}")

    # Cells that hold recall without being credited as `repaired`: the damage reached bytes no
    # cliff region covers, and something else (CRC + EB-fallback) carried them. Worth naming,
    # because "tolerated" reads like the layer did nothing when in fact it repaired its share.
    tol = [r for r in raw if r["arm"] == "on" and r["outcome"] == "tolerated"
           and r.get("cliff_by_region") and sum(r["cliff_by_region"].values()) > 0]
    print(f"  tolerated trials where the cliff layer DID act: {len(tol)}"
          f"   (recall held, but the ex-data CRC still flagged elements, so classify_outcome "
          f"calls it tolerated rather than repaired)")

    report["falsification"] = {
        "repaired_trials": len(repaired),
        "repaired_without_recall_return": len(leaky),
        "protected_crashes_remaining": len(crashes),
        "protected_silent_wrong_remaining": len(silent),
        "repair_attribution": by_region,
        "rotation_regression_clean": rot_ok,
    }

    out = os.path.join(E6, "centroid_gap_summary.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
