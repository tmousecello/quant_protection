"""Phase 3 cost measurement — protection memory + scrub overhead, with a reconciliation harness.

Stage 0 builds the FORMULAS and the analytic-vs-measured reconciliation mechanism; the
concrete numbers (which structures to scrub, how big) are filled in Stage 1 (experiment E1)
once the RaBitQ vulnerability map says where protection must go. Everything here is pure
arithmetic over a region map (the (name -> bytes) breakdown the qp.rabitq adapter produces),
so it is fully unit-testable now on a synthetic region map.

Cost model (Stage 0 brief, component 3):
  mem_cost   = Σ_struct (mult-1) · struct_bytes   +   Σ checksum_bytes
               (replication multiplier `mult` per protected struct; mult=1 => unprotected;
                plus any CRC/checksum bytes added for detection)
  scrub_cost = scrub_bytes · freq                 (bytes re-read per scrub × scrub frequency;
                                                    also expressible as a % of throughput)
  total      = mem_cost + scrub_overhead          (the unified F3 budget)

A `protection_assignment` maps struct name -> {"mult": int, "checksum_bytes": int}. Structs
absent from the assignment are unprotected (mult=1, no checksum), contributing 0 overhead.
"""


def _struct_bytes(region_map):
    """name -> bytes from a region map. Accepts the adapter's {'regions':[{name,byte_len}...]}
    shape or a plain {name: bytes} dict."""
    if isinstance(region_map, dict) and "regions" in region_map:
        return {r["name"]: int(r["byte_len"]) for r in region_map["regions"]}
    return {str(k): int(v) for k, v in region_map.items()}


def mem_cost(region_map, protection_assignment):
    """Σ (mult-1)·struct_bytes + Σ checksum_bytes over protected structs.

    Returns {'replication_bytes':..., 'checksum_bytes':..., 'total_bytes':..., 'by_struct':{}}.
    """
    sizes = _struct_bytes(region_map)
    replication = 0
    checksum = 0
    by_struct = {}
    for name, spec in (protection_assignment or {}).items():
        if name not in sizes:
            raise KeyError(f"protected struct {name!r} not in region map {sorted(sizes)}")
        mult = int(spec.get("mult", 1))
        cks = int(spec.get("checksum_bytes", 0))
        if mult < 1 or cks < 0:
            raise ValueError(f"bad protection for {name!r}: mult>=1, checksum_bytes>=0")
        rep = (mult - 1) * sizes[name]
        replication += rep
        checksum += cks
        by_struct[name] = rep + cks
    return {
        "replication_bytes": replication,
        "checksum_bytes": checksum,
        "total_bytes": replication + checksum,
        "by_struct": by_struct,
    }


def scrub_cost(scrub_bytes, freq):
    """Bytes re-read per unit time by the scrubber: scrub_bytes · freq."""
    if scrub_bytes < 0 or freq < 0:
        raise ValueError("scrub_bytes and freq must be >= 0")
    return float(scrub_bytes) * float(freq)


def scrub_overhead_pct(scrub_bytes, freq, throughput_bytes_per_s):
    """Scrub cost as a percentage of available throughput (the F3-friendly form)."""
    if throughput_bytes_per_s <= 0:
        raise ValueError("throughput must be > 0")
    return 100.0 * scrub_cost(scrub_bytes, freq) / float(throughput_bytes_per_s)


def total_cost(mem, scrub_overhead):
    """Unified F3 budget: protection memory bytes + scrub overhead (same unit as caller passes)."""
    mem_total = mem["total_bytes"] if isinstance(mem, dict) else float(mem)
    return float(mem_total) + float(scrub_overhead)


def reconcile(analytic_protected_bytes, serialized_size_delta, rtol=0.01, atol=0.0):
    """Assert the analytic protection bytes match the measured serialized-size increase.

    The Stage-0 reconciliation gate: the formula's predicted overhead must equal the real
    delta between a protected index's serialized size and the clean baseline's. In Stage 0
    this runs on a synthetic delta; in Stage 1 the delta comes from a real protected RaBitQ
    serialization. Returns a dict; raises AssertionError if they disagree beyond tolerance.
    """
    a = float(analytic_protected_bytes)
    m = float(serialized_size_delta)
    tol = atol + rtol * max(abs(a), abs(m))
    ok = abs(a - m) <= tol
    if not ok:
        raise AssertionError(
            f"cost reconciliation FAILED: analytic={a} measured={m} "
            f"|diff|={abs(a - m)} > tol={tol}")
    return {"analytic": a, "measured": m, "abs_diff": abs(a - m), "tol": tol, "ok": ok}


def monotonicity_check(costs_by_strength):
    """Protection strength ↑ ⇒ cost ↑. costs_by_strength: list of (strength, cost) or cost list.

    Returns True iff cost is non-decreasing in strength order; raises on a violation so a
    mis-specified protection ladder is caught loudly.
    """
    if costs_by_strength and isinstance(costs_by_strength[0], (tuple, list)):
        ordered = [c for _, c in sorted(costs_by_strength, key=lambda x: x[0])]
    else:
        ordered = list(costs_by_strength)
    for i in range(1, len(ordered)):
        if ordered[i] < ordered[i - 1]:
            raise AssertionError(
                f"cost not monotonic in protection strength: {ordered[i]} < {ordered[i-1]} "
                f"at index {i}")
    return True
