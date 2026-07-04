"""Phase 3 Stage 2 — E5: Two-layer recovery mechanism.

Provides RecoveryGuard: cliff layer (rotation majority-vote repair) + slope layer
(ex-data CRC detection + EB-fallback + lazy batch reload) + bounds-check layer (pointer OOB).

Interface contract (§1 of Stage-2 brief):
  init_from_clean(clean_buf)          — snapshot clean state (rotation copies, ex CRCs)
  search_with_recovery(buf, tmp_path) — scrub cliff, detect slope, bounds-check, search
  counters()                          — current counter snapshot (dict)
  scrub_if_due(buf, tick)             — lazy batch reload when slope corruption > threshold

Layer behaviour:
  Cliff (rotation, 64 B, global):
    Maintains R=3 copies. Per-query majority-vote + CRC verify → repair buf + copies.
    Flags irrecoverable (≥ majority copies corrupted same way: CRC of majority ≠ clean).

  Slope (ex_code data, per-chunk CRC-32):
    Per-call CRC check on configurable chunk_size (default 4096 B).
    Fail → EB-fallback (imports qp.rabitq.eb_policy formula; on x86 calls exp_faultinject).
    Optional parity toggle: attempt 1-byte XOR repair before escalating.
    Lazy reload when failed_fraction > reload_threshold.

  Bounds-check (links / cluster_id / label):
    Scans a sample of pointer fields; OOB values logged, element skipped in counts.
    Prevents the crash failure mode (CalledProcessError) for pointer corruption.
"""

import argparse
import json
import os
import tempfile
import zlib

import numpy as np

from qp import config, metrics
from qp.rabitq import layout
from qp.rabitq.registry import get_adapter, adapter_name

# Default configs
SMOKE_CFG = {
    "ticks": 10,
    "R": 3,                    # rotation replica count
    "chunk_size": 4096,        # ex-data chunk size in bytes (CRC granularity)
    "reload_threshold": 0.10,  # lazy-reload trigger fraction
    "parity_on": False,        # parity repair toggle (default off)
    "bounds_sample": 16,       # number of elements to scan for OOB pointers
    "ef": config.EF if hasattr(config, "EF") else 2000,
    "seed": config.SEED,
}
FULL_CFG = {**SMOKE_CFG, "ticks": 100}


class RecoveryGuard:
    """Two-layer + bounds-check recovery guard for a RaBitQ index buffer.

    Instantiate with an adapter, config dict, and the serialized region map.
    Call init_from_clean(clean_buf) once before any corruption.
    Then call search_with_recovery / scrub_if_due in the experiment loop.
    """

    def __init__(self, adapter, cfg, rmap):
        self._adapter = adapter
        self._aname = adapter_name(adapter)
        self._cfg = cfg
        self._rmap = rmap
        hdr = rmap["header"]
        self._cur_element_count = int(hdr["cur_element_count"])

        # Cliff state
        self._rot_region = self._find_region("rotation")    # (byte_start, byte_len)
        self._rot_copies = []
        self._rot_clean_crc = None
        self._cliff_checked = 0
        self._cliff_repaired = 0
        self._cliff_irrecoverable = 0

        # Slope state
        self._ex_region = self._find_region_ex()            # (byte_start, byte_len) or None
        self._chunk_size = int(cfg.get("chunk_size", 4096))
        self._clean_crcs = []
        self._clean_ex_snap = None
        self._parity_bytes = []       # 1-byte XOR parity per chunk (if parity_on)
        self._failed_chunks = set()
        self._slope_checked = 0
        self._slope_failed = 0
        self._slope_reloaded = 0

        # Bounds-check state
        self._oob_elements = 0
        self._clean_buf = None        # clean snapshot, set in init_from_clean (for OOB restore)

        if cfg.get("parity_on", False) and self._chunk_size > 64:
            _log(f"[e5] WARNING: parity_on=True but chunk_size={self._chunk_size} > 64 B; "
                 f"parity single-byte repair only applies to chunks <= 64 B, so it will no-op "
                 f"(every CRC failure escalates straight to EB-fallback).")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def init_from_clean(self, clean_buf):
        """Snapshot rotation copies + CRC, ex-data CRCs + clean snapshot, and full clean buf."""
        self._clean_buf = np.array(clean_buf, dtype=np.uint8)   # authoritative for OOB restore
        R = int(self._cfg.get("R", 3))
        bs, bl = self._rot_region
        rot_bytes = bytes(clean_buf[bs: bs + bl])
        self._rot_copies = [np.frombuffer(rot_bytes, dtype=np.uint8).copy() for _ in range(R)]
        self._rot_clean_crc = zlib.crc32(rot_bytes)

        if self._ex_region is not None:
            exs, exl = self._ex_region
            ex_bytes = clean_buf[exs: exs + exl]
            self._clean_ex_snap = np.array(ex_bytes, dtype=np.uint8)
            self._clean_crcs = self._compute_crcs(ex_bytes)
            if self._cfg.get("parity_on", False):
                self._parity_bytes = self._compute_parity(ex_bytes)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search_with_recovery(self, buf, tmp_path, *, queries=None, k=None, timeout=None):
        """Scrub cliff, check slope, bounds-check, then search.

        Returns {"ids": ndarray(nq,k), "cpp_recall": float, "_eb_path": bool, ...}.
        buf is modified IN PLACE for cliff repairs and lazy reloads.
        """
        k = k or config.K
        ef = int(self._cfg.get("ef", 2000))

        # 1. Cliff: per-query majority-vote repair
        self._scrub_cliff(buf)

        # 2. Slope: per-chunk CRC check
        if self._ex_region is not None:
            self._check_ex_crc(buf)

        # 3. Bounds check: scan sample of pointer fields
        self._bounds_check(buf)

        # 4. Serialize to tmp file
        self._adapter.deserialize_index(buf, tmp_path)

        # 5. Search (EB-fallback path if any slope corruption detected)
        if self._failed_chunks:
            eb_frac = len(self._failed_chunks) / max(1, len(self._clean_crcs))
            seed = int(self._cfg.get("seed", config.SEED))
            res = self._adapter.search_with_eb_fallback(
                tmp_path, eb_frac, seed=seed if self._aname == "real" else 0,
                k=k, ef=ef, out_path=None, timeout=timeout)
        else:
            out_p = tmp_path + ".ids.ivecs"
            res = self._adapter.search_corrupted(
                tmp_path, k=k, ef=ef, out_path=out_p, timeout=timeout)

        return res

    def rotation_replicas(self):
        """The live R rotation copies (mutable arrays) — exposed for --inject-replicas.

        Experiment A's honest caveat: the R=3 copies live in independent memory that the
        default runs never corrupt, so cliff_irrecoverable=0 is structural. Injecting into
        these (same per-bit process, independent streams) measures majority-vote under
        simultaneous replica accumulation — the multi-copy failure boundary.
        """
        return self._rot_copies

    def scrub_if_due(self, buf, tick):
        """Lazy batch reload: if slope corruption fraction > threshold, reload clean ex chunks."""
        if self._ex_region is None:
            return
        if not self._clean_crcs:
            return
        frac = len(self._failed_chunks) / max(1, len(self._clean_crcs))
        threshold = float(self._cfg.get("reload_threshold", 0.10))
        if frac > threshold:
            self._reload_slope(buf)

    def counters(self):
        """Current counter snapshot (schema as per §1 contract)."""
        total_chunks = max(1, len(self._clean_crcs))
        return {
            "cliff_checked": self._cliff_checked,
            "cliff_repaired": self._cliff_repaired,
            "cliff_irrecoverable": self._cliff_irrecoverable,
            "slope_checked": self._slope_checked,
            "slope_failed": self._slope_failed,
            "slope_reloaded": self._slope_reloaded,
            "known_corrupted": len(self._failed_chunks),
            "eb_fraction": len(self._failed_chunks) / total_chunks,
            "oob_elements": self._oob_elements,
        }

    # ------------------------------------------------------------------
    # Cliff layer internals
    # ------------------------------------------------------------------

    def _scrub_cliff(self, buf):
        """Majority-vote across R rotation copies; repair copies + buf; verify CRC."""
        R = len(self._rot_copies)
        if R == 0:
            return
        bs, bl = self._rot_region
        majority = np.zeros(bl, dtype=np.uint8)

        repaired_bits = 0
        for i in range(bl):
            for j in range(8):
                votes = sum(1 for r in range(R) if (self._rot_copies[r][i] >> j) & 1)
                bit_val = 1 if votes > R // 2 else 0
                if bit_val:
                    majority[i] |= (1 << j)
                # Count bits that differ from majority (across all copies + buf)
                for r in range(R):
                    if ((self._rot_copies[r][i] >> j) & 1) != bit_val:
                        repaired_bits += 1
                buf_bit = (buf[bs + i] >> j) & 1
                if buf_bit != bit_val:
                    repaired_bits += 1
        self._cliff_checked += bl * 8

        # Verify the majority-vote result against the authoritative clean CRC BEFORE writing it.
        # If it fails, >= majority copies were corrupted the same way: flag irrecoverable and do
        # NOT write the wrong majority into buf (plan §5a: detect, never silently serve a wrong
        # value). buf is left as-is for the search to proceed under the loud failure flag.
        if zlib.crc32(bytes(majority)) != self._rot_clean_crc:
            self._cliff_irrecoverable += 1
            _log(f"[e5] IRRECOVERABLE: majority-vote rotation result fails clean CRC "
                 f"(≥{(R+1)//2} copies corrupted same way); buf left unrepaired, not served as fixed")
            return

        if repaired_bits:
            self._cliff_repaired += repaired_bits
            buf[bs: bs + bl] = majority
            for r in range(R):
                np.copyto(self._rot_copies[r], majority)

    # ------------------------------------------------------------------
    # Slope layer internals
    # ------------------------------------------------------------------

    def _chunks(self):
        """Yield (chunk_idx, abs_start, abs_end) for each ex-data chunk."""
        if self._ex_region is None:
            return
        exs, exl = self._ex_region
        sz = self._chunk_size
        for i in range(0, (exl + sz - 1) // sz):
            start = exs + i * sz
            end = min(exs + exl, start + sz)
            yield i, start, end

    def _compute_crcs(self, ex_bytes):
        """CRC-32 per chunk of ex_bytes (a slice of buf starting at byte 0 of the region)."""
        sz = self._chunk_size
        exl = len(ex_bytes)
        crcs = []
        for i in range(0, (exl + sz - 1) // sz):
            chunk = bytes(ex_bytes[i * sz: min(exl, (i + 1) * sz)])
            crcs.append(zlib.crc32(chunk))
        return crcs

    def _compute_parity(self, ex_bytes):
        """1-byte XOR parity per chunk."""
        sz = self._chunk_size
        exl = len(ex_bytes)
        parity = []
        for i in range(0, (exl + sz - 1) // sz):
            chunk = ex_bytes[i * sz: min(exl, (i + 1) * sz)]
            p = 0
            for b in chunk:
                p ^= int(b)
            parity.append(p)
        return parity

    def _check_ex_crc(self, buf):
        """CRC-32 check per chunk; failed chunks enter _failed_chunks set."""
        parity_on = self._cfg.get("parity_on", False)
        exs, _ = self._ex_region

        for chunk_idx, abs_start, abs_end in self._chunks():
            chunk = buf[abs_start:abs_end]
            self._slope_checked += 1
            crc = zlib.crc32(bytes(chunk))
            if crc != self._clean_crcs[chunk_idx]:
                if parity_on and chunk_idx < len(self._parity_bytes):
                    repaired = self._try_parity_repair(chunk, chunk_idx)
                    if repaired is not None:
                        buf[abs_start:abs_end] = repaired
                        self._failed_chunks.discard(chunk_idx)
                        continue
                if chunk_idx not in self._failed_chunks:
                    self._slope_failed += 1   # count distinct failure EVENTS, not per-tick repeats
                self._failed_chunks.add(chunk_idx)
            else:
                self._failed_chunks.discard(chunk_idx)

    def _try_parity_repair(self, chunk, chunk_idx):
        """XOR-parity single-byte correction: if exactly one byte wrong, fix it.

        Returns the repaired bytes on success, None if uncorrectable (parity can only
        locate a single-bit change per byte if the XOR parity of the whole chunk is
        non-zero AND only one byte is different — we test this by brute-force over small
        chunks; if chunk is large, fall back to None).
        """
        stored_p = self._parity_bytes[chunk_idx]
        exs, _ = self._ex_region
        local = np.array(chunk, dtype=np.uint8)
        actual_p = 0
        for b in local:
            actual_p ^= int(b)
        if actual_p == stored_p:
            return None   # even number of errors or no error but CRC still failed (collision unlikely)
        # XOR parity tells us the XOR of all flipped bytes. If exactly one byte is wrong
        # by a single bit, we can identify it. For robustness, attempt brute-force correction
        # only if the chunk is small (≤ 64 B); larger chunks use EB-fallback directly.
        if len(local) > 64:
            return None
        diff_p = actual_p ^ stored_p   # XOR of all changes
        for i, b in enumerate(local):
            candidate = local.copy()
            candidate[i] = b ^ diff_p
            cp = 0
            for x in candidate:
                cp ^= int(x)
            if cp == stored_p and zlib.crc32(bytes(candidate)) == self._clean_crcs[chunk_idx]:
                return candidate
        return None

    def _reload_slope(self, buf):
        """Reload corrupted ex chunks from the clean snapshot."""
        if self._clean_ex_snap is None or not self._failed_chunks:
            return
        exs, exl = self._ex_region
        sz = self._chunk_size
        reloaded = 0
        for chunk_idx in list(self._failed_chunks):
            local_start = chunk_idx * sz
            local_end = min(exl, (chunk_idx + 1) * sz)
            abs_start = exs + local_start
            abs_end = exs + local_end
            buf[abs_start:abs_end] = self._clean_ex_snap[local_start:local_end]
            reloaded += 1
        self._failed_chunks.clear()
        self._slope_reloaded += reloaded
        # Do NOT recompute _clean_crcs from buf: they already hold the authoritative clean CRCs,
        # and the reloaded chunks now match them. Recomputing from buf would bake a CRC-collision
        # dirty chunk (one that slipped past detection) in as the new "clean" baseline.
        _log(f"[e5] slope reload: restored {reloaded} chunks, counter reset")

    # ------------------------------------------------------------------
    # Bounds-check layer
    # ------------------------------------------------------------------

    def _bounds_check(self, buf):
        """Scan element link/cluster_id/label fields for OOB pointers; restore from clean on a hit.

        Detection scans a FIXED slot window (maxM0), NOT the on-disk neighbour count — the count
        itself may be corrupted to 0, which would otherwise hide every OOB id behind it. On a hit
        the whole field is restored from the clean snapshot (faithful "skip the bad edge"); this
        both prevents the downstream crash (plan §5c "不 crash") and never loses the structure.
        """
        n_sample = int(self._cfg.get("bounds_sample", 16))
        n_elem = self._cur_element_count
        if n_elem == 0:
            return
        hdr = self._rmap["header"]
        num_cluster = int(hdr["num_cluster"])
        maxM0 = int(hdr["maxM0"])
        sample_ids = list(range(min(n_sample, n_elem)))
        for e in sample_ids:
            for field in ("links", "cluster_id", "label"):
                try:
                    bs, bl = layout.element_field_range(self._rmap, field, e)
                except (KeyError, IndexError):
                    continue
                oob = False
                if field == "links":
                    # links: [uint32 count][maxM0 * uint32 ids]. Scan all physically-present slots.
                    if bl < 8:
                        continue
                    n_slots = min(maxM0, (bl - 4) // 4)
                    for i in range(n_slots):
                        vid = int.from_bytes(bytes(buf[bs + 4 + i * 4: bs + 8 + i * 4]), "little")
                        if vid >= n_elem and vid != 0xFFFFFFFF:
                            oob = True
                            break
                else:
                    if bl < 4:
                        continue
                    limit = num_cluster if field == "cluster_id" else n_elem
                    val = int.from_bytes(bytes(buf[bs: bs + 4]), "little")
                    if val >= limit and val != 0xFFFFFFFF:
                        oob = True
                if oob:
                    self._oob_elements += 1
                    if self._clean_buf is not None:
                        buf[bs: bs + bl] = self._clean_buf[bs: bs + bl]   # restore -> no crash
                    _log(f"[e5] OOB {field}: elem={e} restored from clean snapshot; skip")

    # ------------------------------------------------------------------
    # Region helpers
    # ------------------------------------------------------------------

    def _find_region(self, name):
        """Find (byte_start, byte_len) for a global region by name."""
        for r in self._rmap["regions"]:
            if layout.base_name(r["name"]) == name and r.get("byte_start") is not None:
                return (int(r["byte_start"]), int(r["byte_len"]))
        raise KeyError(f"region {name!r} not found in rmap")

    def _find_region_ex(self):
        """Element-0 ex_code as the representative slope region, or None if no ex data.

        The slope layer protects the ex_data STRUCTURE; using element 0's ex_code (96 B) — not
        the whole level0 span the previous code computed — keeps the studied variable isolated
        and matches the E3a/E3b element-0 convention. Experiment B's multi-element ex gradient
        is a separate workstation concern.
        """
        if int(self._rmap["header"]["cur_element_count"]) == 0:
            return None
        try:
            return layout.element_field_range(self._rmap, "ex_code", 0)
        except (KeyError, IndexError):
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg):
    print(msg)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(args):
    cfg = SMOKE_CFG.copy() if args.smoke else FULL_CFG.copy()
    if args.ticks:
        cfg["ticks"] = int(args.ticks)
    cfg["seed"] = args.seed
    cfg["parity_on"] = args.parity_on

    out_dir = os.path.join(config.ROOT, "artifacts_smoke" if args.smoke else "artifacts",
                           "phase3", "e5")
    os.makedirs(out_dir, exist_ok=True)

    from phase3_e3c_temporal import TemporalCorruptor, _resolve_region, region_label

    adapter = get_adapter(args.adapter)
    aname = adapter_name(adapter)

    clean_buf = adapter.serialize_index()
    rmap = adapter.region_map()
    gt = adapter.load_groundtruth()

    guard = RecoveryGuard(adapter, cfg, rmap)
    guard.init_from_clean(clean_buf)

    # Set up E3c for the specified region + pattern
    region_name = args.region
    rlabel = region_label(region_name)
    region = _resolve_region(rmap, region_name)
    corruptor = TemporalCorruptor(region, cfg)

    buf = clean_buf.copy()

    # --inject-replicas: subject each rotation replica to the SAME per-tick fault process
    # (same p / pattern cfg) on an INDEPENDENT stream (seed differs per replica), closing the
    # "replicas are immortal" caveat. Each replica is a standalone buffer -> region (0, len).
    inject_replicas = bool(getattr(args, "inject_replicas", False))
    rep_corruptors = []
    if inject_replicas:
        rep_corruptors = [TemporalCorruptor((0, int(copy.size)), cfg)
                          for copy in guard.rotation_replicas()]

    print(f"[e5] adapter={aname} region={rlabel} pattern={args.pattern} "
          f"ticks={cfg['ticks']} parity_on={cfg['parity_on']} "
          f"inject_replicas={inject_replicas}")

    records = []
    # --inject-replicas runs get their own filenames — never clobber the baseline records.
    stem = f"e5_{args.pattern}_{rlabel}" + ("_replicas" if inject_replicas else "")
    raw_path = os.path.join(out_dir, f"{stem}.records.jsonl")
    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp_f:
        tmp_path = tmp_f.name

    try:
        with open(raw_path, "w") as raw_f:
            for tick in range(cfg["ticks"]):
                seed = cfg["seed"] ^ tick
                corruptor.inject_step(buf, args.pattern, seed)
                cc = corruptor.cumulative_corruption()

                if inject_replicas:
                    for r, (rc, copy) in enumerate(zip(rep_corruptors,
                                                       guard.rotation_replicas())):
                        # independent stream per replica: same process, different seed lane
                        rc.inject_step(copy, args.pattern, seed ^ ((r + 1) * 0x5EED))

                res = guard.search_with_recovery(buf, tmp_path)
                ids = res.get("ids")
                recall = float(metrics.recall_at_k(ids, gt, config.K)) if ids is not None else None

                ctr = guard.counters()
                row = {"tick": tick, "cumulative_corruption": cc,
                       "recall@10": recall, "counters": ctr,
                       "_eb_path": res.get("_eb_path", False),
                       "pattern": args.pattern, "region": rlabel, "adapter": aname}
                if inject_replicas:
                    # bits currently differing from clean, per replica (post-scrub state)
                    row["replica_bits"] = [rc.cumulative_corruption()["bits_flipped"]
                                           for rc in rep_corruptors]
                raw_f.write(json.dumps(row) + "\n")
                records.append(row)
                guard.scrub_if_due(buf, tick)

                if tick % 5 == 0 or tick == cfg["ticks"] - 1:
                    print(f"  tick={tick:3d}  bits={cc['bits_flipped']:4d}  "
                          f"recall={recall:.4f}  cliff_repaired={ctr['cliff_repaired']}  "
                          f"slope_failed={ctr['slope_failed']}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    summary = {
        "pattern": args.pattern, "region": rlabel, "ticks": cfg["ticks"],
        "final_counters": records[-1]["counters"] if records else {},
        "final_recall@10": records[-1]["recall@10"] if records else None,
        "adapter": aname, "cfg": cfg,
    }
    if inject_replicas:
        summary["inject_replicas"] = True
        summary["final_replica_bits"] = records[-1].get("replica_bits") if records else None
    out_json = os.path.join(out_dir, f"{stem}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[e5] done → {out_json}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="E5: two-layer recovery mechanism runner")
    ap.add_argument("--adapter", default="auto", choices=("stub", "real", "auto"))
    ap.add_argument("--region", default="rotation")
    ap.add_argument("--pattern", default="uniform_accum")
    ap.add_argument("--ticks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--parity-on", dest="parity_on", action="store_true")
    ap.add_argument("--inject-replicas", dest="inject_replicas", action="store_true",
                    help="also corrupt the R=3 rotation replicas each tick (same process, "
                         "independent streams) — measures majority-vote failure boundary")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
