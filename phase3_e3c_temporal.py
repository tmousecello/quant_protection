"""Phase 3 Stage 2 — E3c: Temporal cumulative fault model.

Provides TemporalCorruptor: a timeline-driven accumulator that injects corruption into
a uint8 buffer tick by tick, accumulating (never restoring) until reset() is called.
Wraps qp.faults injectors for position selection; maintains XOR physical state so that
a bit flipped twice cancels correctly (net effect = clean).

Interface contract (§1 of Stage-2 brief):
  inject_step(buf, pattern, seed) — inject one tick of corruption (cumulative, no restore)
  cumulative_corruption()         — current {"bits_flipped", "fraction", "tick"}
  reset(buf)                      — restore buf to clean state, clear accumulator

Patterns:
  uniform_accum   — each tick: Binomial(n_bits, p) uniform random flips (qp.faults.uniform_p)
  clustered_accum — each tick: spatial_cluster(W, k, n_win) window flips
  cross_row_accum — each tick: cross_row(stride) pair flips
  burst_accum     — quiet ticks + burst at configured tick (qp.faults.temporal_burst)

All patterns are deterministic in the per-tick seed the caller supplies.
"""

import argparse
import json
import os
import tempfile

import numpy as np

from qp import config, faults
from qp.rabitq import layout
from qp.rabitq.registry import get_adapter, adapter_name

# Supported pattern names
PATTERNS = ("uniform_accum", "clustered_accum", "cross_row_accum", "burst_accum")

# Default config (overridden per experiment)
SMOKE_CFG = {
    "ticks": 10,
    "p": 0.005,               # uniform_accum per-bit probability
    "W": 32,                  # clustered: window width in bits
    "k": 4,                   # clustered: flips per window
    "n_win": 2,               # clustered: windows per tick
    "stride": 8,              # cross_row: stride in bytes
    "burst_tick": 5,          # burst_accum: trigger tick
    "burst_n": 20,            # burst_accum: burst size in bits
    "region": "rotation",
    "seed": config.SEED,
}
FULL_CFG = {**SMOKE_CFG, "ticks": 100}


class TemporalCorruptor:
    """Timeline-driven cumulative fault injector.

    Operates on any uint8 buffer + a byte region (byte_start, byte_len).
    Each inject_step call selects positions via the configured pattern and XOR-toggles
    them in the buffer and in the internal dirty-bits set. Double-toggle of the same bit
    cancels (clean XOR state), so cumulative_corruption() tracks the net physical error count.
    """

    def __init__(self, region, cfg):
        """
        region : (byte_start, byte_len) tuple
        cfg    : dict with pattern-specific keys (see SMOKE_CFG for reference)
        """
        self._byte_start = int(region[0])
        self._byte_len = int(region[1])
        self._n_bits = self._byte_len * 8
        self._cfg = cfg
        self._dirty_bits = set()   # region-relative bit offsets currently corrupted (XOR state)
        self._tick = 0
        self._curve = []           # cumulative bit count per tick

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def inject_step(self, buf, pattern, seed):
        """Inject one tick of corruption and accumulate.

        Calls the qp.faults injector for pattern on a scratch buffer to get positions,
        then XOR-toggles each position in buf and updates the dirty-bits set.
        Returns the list of (byte, bit) positions toggled this tick.
        """
        if pattern not in PATTERNS:
            raise ValueError(f"unknown pattern {pattern!r}; choose from {PATTERNS}")
        positions = self._sample_positions(pattern, seed)
        self._toggle(buf, positions)
        self._tick += 1
        self._curve.append(len(self._dirty_bits))
        return positions

    def cumulative_corruption(self):
        """Current accumulation state: bits currently corrupted (physical XOR count)."""
        return {
            "bits_flipped": len(self._dirty_bits),
            "fraction": len(self._dirty_bits) / self._n_bits if self._n_bits else 0.0,
            "tick": self._tick,
        }

    def reset(self, buf):
        """Restore buf to clean state and clear accumulator (models a full scrub)."""
        for rel_off in self._dirty_bits:
            byte_abs = self._byte_start + rel_off // 8
            bit = rel_off % 8
            buf[byte_abs] ^= (1 << bit)
        self._dirty_bits.clear()
        self._tick = 0
        self._curve.clear()

    def cumulative_curve(self):
        """Running corrupted-bit count per tick (list of ints, one per inject_step call)."""
        return list(self._curve)

    # ------------------------------------------------------------------
    # Internal: position selection and XOR-toggle
    # ------------------------------------------------------------------

    def _scratch_region(self):
        """Zero buffer + region starting at byte 0 (for scratch-based injector calls)."""
        scratch = np.zeros(self._byte_len, dtype=np.uint8)
        region = (0, self._byte_len)
        return scratch, region

    def _sample_positions(self, pattern, seed):
        """Call the appropriate qp.faults injector on a scratch buffer; shift to absolute coords."""
        cfg = self._cfg
        scratch, region = self._scratch_region()

        if pattern == "uniform_accum":
            p = float(cfg.get("p", 0.005))
            raw = faults.uniform_p(scratch, region, p, seed)

        elif pattern == "clustered_accum":
            W = int(cfg.get("W", 32))
            k = int(cfg.get("k", 4))
            n_win = int(cfg.get("n_win", 2))
            raw = faults.spatial_cluster(scratch, region, W, k, n_win, seed)

        elif pattern == "cross_row_accum":
            stride = int(cfg.get("stride", 8))
            raw = faults.cross_row(scratch, region, stride, seed)

        elif pattern == "burst_accum":
            burst_tick = int(cfg.get("burst_tick", 5))
            burst_n = int(cfg.get("burst_n", 20))
            if self._tick != burst_tick:
                return []   # quiet tick
            timeline = [0] * burst_tick + [burst_n]
            raw = faults.temporal_burst(scratch, region, timeline, seed)

        else:
            raise AssertionError(f"unhandled pattern {pattern!r}")

        # Shift scratch-relative (byte, bit) to absolute coords
        return [(byte_rel + self._byte_start, bit) for (byte_rel, bit) in raw]

    def _toggle(self, buf, positions):
        """XOR-toggle positions in buf and update _dirty_bits (double-toggle = cancel)."""
        for (byte_abs, bit) in positions:
            rel = (byte_abs - self._byte_start) * 8 + bit
            if rel in self._dirty_bits:
                self._dirty_bits.discard(rel)
            else:
                self._dirty_bits.add(rel)
            buf[byte_abs] ^= (1 << bit)


# ---------------------------------------------------------------------------
# Runner (smoke / full, config-driven, adapter-injected)
# ---------------------------------------------------------------------------

def _resolve_region(rmap, region_name):
    """Extract (byte_start, byte_len) for a named region from the serialized region map."""
    for r in rmap["regions"]:
        if layout.base_name(r["name"]) == region_name and r.get("byte_start") is not None:
            return (int(r["byte_start"]), int(r["byte_len"]))
    # Per-element region: sum all element offsets (aggregate)
    agg = layout.aggregate_region_map(rmap)
    if region_name in agg:
        raise ValueError(
            f"region {region_name!r} is per-element (no single byte_start); "
            f"use a global region like 'rotation' for E3c timeline experiments.")
    raise KeyError(f"region {region_name!r} not found in region map")


def run(args):
    cfg = SMOKE_CFG.copy() if args.smoke else FULL_CFG.copy()
    if args.ticks:
        cfg["ticks"] = int(args.ticks)
    if args.p:
        cfg["p"] = float(args.p)
    cfg["region"] = args.region
    cfg["seed"] = args.seed

    out_dir = os.path.join(config.ROOT, "artifacts_smoke" if args.smoke else "artifacts",
                           "phase3", "e3c")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"e3c_{args.pattern}_{args.region}.records.jsonl")

    adapter = get_adapter(args.adapter)
    aname = adapter_name(adapter)

    buf = adapter.serialize_index()
    rmap = adapter.region_map()
    region = _resolve_region(rmap, cfg["region"])

    corruptor = TemporalCorruptor(region, cfg)

    records = []
    print(f"[e3c] adapter={aname} region={cfg['region']} pattern={args.pattern} "
          f"ticks={cfg['ticks']}")

    with open(raw_path, "w") as raw_f:
        for tick in range(cfg["ticks"]):
            seed = cfg["seed"] ^ tick
            positions = corruptor.inject_step(buf, args.pattern, seed)
            cc = corruptor.cumulative_corruption()
            row = {"tick": tick, "positions_this_tick": len(positions),
                   "cumulative_corruption": cc, "pattern": args.pattern,
                   "region": cfg["region"], "adapter": aname}
            raw_f.write(json.dumps(row) + "\n")
            records.append(row)
            if tick % 10 == 0 or tick == cfg["ticks"] - 1:
                print(f"  tick={tick:3d}  bits_flipped={cc['bits_flipped']:4d}  "
                      f"fraction={cc['fraction']:.4f}")

    # Reset check: verify reset restores buf
    clean_check = adapter.serialize_index()
    corruptor.reset(buf)
    drift = int(np.sum(buf != clean_check))
    assert drift == 0, f"[e3c] reset() drift = {drift} bytes (bug in XOR toggle)"
    print(f"[e3c] reset OK (drift=0)")

    summary = {
        "pattern": args.pattern,
        "region": cfg["region"],
        "ticks": cfg["ticks"],
        "final_bits_flipped": records[-1]["cumulative_corruption"]["bits_flipped"],
        "final_fraction": records[-1]["cumulative_corruption"]["fraction"],
        "adapter": aname,
        "cfg": cfg,
    }
    out_json = os.path.join(out_dir, f"e3c_{args.pattern}_{args.region}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[e3c] done → {out_json}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="E3c: temporal cumulative fault model runner")
    ap.add_argument("--adapter", default="auto", choices=("stub", "real", "auto"))
    ap.add_argument("--region", default="rotation",
                    help="region name (e.g. rotation, ex_code)")
    ap.add_argument("--pattern", default="uniform_accum", choices=PATTERNS)
    ap.add_argument("--ticks", type=int, default=None)
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
