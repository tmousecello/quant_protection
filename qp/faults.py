"""Fault models — the three memory-error injectors for Phase 3 (and reusable beyond it).

These sit on top of the pure-numpy bit primitives in qp.bits (so this module imports
faiss-free) and are the single source of truth for *how* corruption is placed. They are
deliberately index-agnostic: each operates on a writable uint8 buffer and a ``region``
``(byte_start, byte_len)`` window, so the same injector drives FAISS indexes (Phase 1/2),
RaBitQ index files (Phase 3, via qp.rabitq), or any synthetic buffer (the unit tests).

The three models map to distinct hardware error mechanisms (prompt.txt / Stage 0 brief):

1. ``uniform_p``        — spatially-uncorrelated SEU / retention failure: each bit in the
                          region flips iid with probability ``p`` (p ∈ 1e-6 … 1e-3).
2. ``spatial_cluster``  /
   ``cross_row``        — rowhammer / retention clustering: flips concentrate inside small
                          windows, or hit the same offset across adjacent rows.
3. ``temporal_burst``   — aging / thermal hotspots: a quiet span then a burst of flips at a
                          trigger; ``CumulativeCorruption`` accumulates bursts across ticks
                          for the Stage-1 scrub-interval study.

Every injector is deterministic in ``seed``, returns the exact list of flipped
``(byte, bit)`` positions, and is undone byte-for-byte by ``restore`` (re-XOR). That makes
inject → restore round-trips identity, which the verifications assert.
"""
import numpy as np

from qp.bits import flip_bits


# --- region helpers ----------------------------------------------------------

def _region_bounds(region):
    """(byte_start, byte_len) -> (lo_byte, n_bits). Raises on a malformed region."""
    byte_start, byte_len = int(region[0]), int(region[1])
    if byte_start < 0 or byte_len <= 0:
        raise ValueError(f"bad region {region!r}: need byte_start>=0, byte_len>0")
    return byte_start, byte_len * 8


def _bit_to_pos(byte_start, bit_in_region):
    """Absolute (byte_pos, bit) for a bit offset measured from the region's first bit."""
    abs_bit = byte_start * 8 + int(bit_in_region)
    return (abs_bit // 8, abs_bit % 8)


def restore(buf, positions):
    """Undo an injection: re-XOR the exact (byte, bit) positions. Identity round-trip.

    Self-inverse because every flip is an XOR — restore(buf, inject(...)) leaves buf
    byte-for-byte identical to before the inject.
    """
    flip_bits(buf, positions)


# --- model 1: uniform random multi-bit flip ----------------------------------

def uniform_p(buf, region, p, seed):
    """Flip each bit in ``region`` independently with probability ``p``.

    Models spatially-uncorrelated errors (cosmic-ray SEU, retention failure). ``p`` is the
    per-bit error rate (the brief sweeps 1e-6 … 1e-3). Returns the flipped (byte, bit)
    positions; pass them to ``restore`` to undo. Deterministic in ``seed``: the same
    (region, p, seed) always selects the same positions.

    Expected flip count is N·p with N = region bits, so verification checks the realized
    count against a binomial CI and the position histogram against uniform (chi-square).
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0,1], got {p}")
    byte_start, n_bits = _region_bounds(region)
    rng = np.random.default_rng(seed)
    # Memory-safe: draw the flip COUNT from Binomial(n_bits, p), then sample that many distinct
    # bit offsets — instead of materializing one float per region bit. A whole-index region is
    # ~1e9+ bits, so rng.random(n_bits) would allocate a multi-GB float array. O(count) memory,
    # deterministic in seed. (Trade-off: the flipped set is an iid sample of size Binomial(N,p),
    # so it is NOT nested across p the way the old per-bit threshold method was.)
    count = int(rng.binomial(n_bits, p))
    bit_offsets = rng.choice(n_bits, size=count, replace=False)
    positions = [_bit_to_pos(byte_start, int(b)) for b in bit_offsets]
    flip_bits(buf, positions)
    return positions


# --- model 2: spatially clustered multi-bit flip -----------------------------

def spatial_cluster(buf, region, W, k, n_win, seed):
    """``n_win`` windows of ``W`` consecutive bits, each with exactly ``k`` flips.

    Models rowhammer / retention clustering: errors are not spread uniformly but bunch
    inside small windows (e.g. 2–3 flips within an 8-bit window). Window start bits are
    chosen at random within the region (non-overlapping when they fit); inside each window
    ``k`` distinct bits are flipped. Returns flipped (byte, bit) positions.

    Verification: exactly k flips per window; the realized positions are demonstrably
    clustered (per-window density far above the uniform expectation); deterministic; and
    restore round-trips.
    """
    byte_start, n_bits = _region_bounds(region)
    if W <= 0 or k <= 0 or k > W:
        raise ValueError(f"need 0<k<=W and W>0, got W={W}, k={k}")
    if n_win <= 0 or n_win * W > n_bits:
        raise ValueError(f"{n_win} windows of {W} bits exceed region ({n_bits} bits)")
    rng = np.random.default_rng(seed)
    # Non-overlapping windows: partition the region into W-sized slots, pick n_win of them.
    n_slots = n_bits // W
    slots = rng.choice(n_slots, size=n_win, replace=False)
    positions = []
    for slot in slots:
        base = int(slot) * W
        within = rng.choice(W, size=k, replace=False)   # k distinct bits in this window
        for w in within:
            positions.append(_bit_to_pos(byte_start, base + int(w)))
    flip_bits(buf, positions)
    return positions


def cross_row(buf, region, stride, seed):
    """Flip the same in-row bit offset across two adjacent rows ``stride`` bytes apart.

    Models the rowhammer victim pattern where a disturbed cell corrupts the corresponding
    bit in a physically adjacent row. One random (byte, bit) is chosen in the region's first
    row; the same in-row offset is flipped again ``stride`` bytes later. Returns both
    positions (so restore undoes the pair). Deterministic in ``seed``.
    """
    byte_start, n_bits = _region_bounds(region)         # validates byte_start>=0, byte_len>0
    byte_len = n_bits // 8
    if stride <= 0:
        raise ValueError(f"stride must be >0, got {stride}")
    if 2 * stride > byte_len:
        raise ValueError(f"region too small ({byte_len} B) for two rows {stride} B apart")
    rng = np.random.default_rng(seed)
    off = int(rng.integers(0, stride))                  # byte offset within the first row
    bit = int(rng.integers(0, 8))
    positions = [(byte_start + off, bit), (byte_start + off + stride, bit)]
    flip_bits(buf, positions)
    return positions


# --- DRAM fault shapes (CIDR spec §B: single_cell / device_row / device_column) ---
#
# These three shapes are the hardware-fault-mode taxonomy from the CIDR submission's §B
# (DDR4 corrigendum, arXiv:2408.15302 Table I): most real DRAM errors are single-bit
# (single_cell), a minority hit an entire physical row (device_row — e.g. a failed row
# decoder / retention-weak row), and a smaller minority hit one bit lane down a physical
# column across many rows (device_column — e.g. a bad I/O gate or TSV in 3D-stacked DRAM).
# `shape_weights.json` (results/jonathan-vuln-shapes/) gives the vendor-reported mix used to
# weight draws across the three.
#
# IMPORTANT MODELING CAVEAT: the 8192-byte "row" used below is OUR modeling unit, chosen so
# a row is large relative to a RaBitQ element (272 B) without being unwieldy — it is NOT a
# datasheet row size. Real DRAM row (page) sizes vary by device/config (typically 1-16 KiB
# of *cells*, which is a different granularity than the serialized-index byte layout these
# injectors operate on to begin with). Getting the true row/column geometry right would
# require device-level characterization (DRAMScope, Nam et al., ISCA 2024; X-ray, IEEE CAL
# 2023) that this project does not have. Treat `row_bytes` as a sensitivity-study knob, not a
# hardware fact.


def single_cell(buf, region, seed):
    """Flip exactly one uniformly-random bit within ``region`` (the baseline DRAM fault shape).

    The dominant real-world case (vendor A: ~78% of reported errors per shape_weights.json):
    a single bit-cell fails independently of its neighbors. Delegates straight to
    ``_bit_to_pos`` — this is ``uniform_p`` with a fixed count of 1. Returns
    ``(positions, record)``; deterministic in ``seed``; ``restore(buf, positions)`` round-trips.
    """
    byte_start, n_bits = _region_bounds(region)
    rng = np.random.default_rng(seed)
    bit_offset = int(rng.integers(0, n_bits))
    pos = _bit_to_pos(byte_start, bit_offset)
    positions = [pos]
    flip_bits(buf, positions)
    byte, _bit = pos
    record = {
        "shape": "single_cell",
        "anchor_byte": int(byte),
        "coverage": [int(byte), int(byte) + 1],
        "bits_flipped": 1,
    }
    return positions, record


def device_row(buf, region, seed, *, row_bytes=8192, p_in_row=0.5):
    """Fail an entire physical DRAM row: every bit in one ``row_bytes``-aligned block flips iid.

    Models a failed row decoder / retention-weak row (vendor A: ~9.7% of reported errors).
    The anchor block is the ``row_bytes``-aligned block (aligned to the BUFFER start, not the
    region) that contains a byte drawn uniformly from ``region``; every bit in that block then
    flips independently with probability ``p_in_row`` (default 0.5 — a fully-failed row, not a
    partial one). The block is clamped to ``[0, len(buf))``, so it commonly extends outside
    ``region`` by design: RaBitQ's level0 fields (links, codes, factors) interleave at 272
    B/element, so an 8 KB row physically spans roughly 30 elements' worth of unrelated fields.
    ``record["coverage"]`` reports the actual clamped ``[lo, hi)`` byte range hit, and
    ``record["bits_flipped"]`` is drawn from ``Binomial((hi-lo)*8, p_in_row)`` via the same
    count-then-choice memory-safe pattern as ``uniform_p`` (draw the flip COUNT, then sample
    that many distinct bit offsets, instead of materializing one float per row bit).

    NOTE on ``row_bytes=8192``: this is our modeling unit, not a datasheet row size — see the
    module-level caveat above (DRAMScope / X-ray citations) for why true device geometry is out
    of scope here.
    """
    byte_start, n_bits = _region_bounds(region)
    byte_len = n_bits // 8
    if row_bytes <= 0:
        raise ValueError(f"row_bytes must be >0, got {row_bytes}")
    if not (0.0 <= p_in_row <= 1.0):
        raise ValueError(f"p_in_row must be in [0,1], got {p_in_row}")
    rng = np.random.default_rng(seed)
    anchor_byte = byte_start + int(rng.integers(0, byte_len))
    lo = (anchor_byte // row_bytes) * row_bytes
    hi = min(lo + row_bytes, len(buf))
    row_bits = (hi - lo) * 8
    count = int(rng.binomial(row_bits, p_in_row))
    bit_offsets = rng.choice(row_bits, size=count, replace=False)
    positions = [_bit_to_pos(lo, int(b)) for b in bit_offsets]
    flip_bits(buf, positions)
    record = {
        "shape": "device_row",
        "anchor_byte": int(anchor_byte),
        "coverage": [int(lo), int(hi)],
        "bits_flipped": count,
        "row_bytes": int(row_bytes),
        "p_in_row": float(p_in_row),
    }
    return positions, record


def device_column(buf, region, seed, *, row_bytes=8192, n_rows=512):
    """Fail one bit lane down a physical DRAM column across ``n_rows`` consecutive rows.

    Models a bad I/O gate / TSV fault in 3D-stacked DRAM (vendor A: ~12.2% of reported errors)
    — the same (byte-offset-in-row, bit) pair fails in every row of a physical column. The
    first row is the ``row_bytes``-aligned block (aligned to the buffer start) containing a
    byte drawn uniformly from ``region``. Within that row a ``(col_offset, col_bit)`` pair is
    then chosen uniformly from the subset of ``[0, row_bytes) x [0, 8)`` whose hit byte
    (``first_row_start + col_offset``) lies inside ``region`` — this is always non-empty since
    the anchor byte itself qualifies. That exact ``(col_offset, col_bit)`` is then flipped in
    ``n_rows`` consecutive ``row_bytes``-aligned blocks starting at the first row, clamped at
    ``len(buf)`` (a stripe that runs off the end of the buffer is simply shorter). Because 8192
    is not a multiple of RaBitQ's 272 B/element stride, the stripe precesses across different
    per-element fields row to row rather than always hitting e.g. the same code byte — so
    ``positions`` records every hit's absolute ``(byte, bit)``, not just the pattern.

    NOTE: ``row_bytes=8192`` is our modeling unit, not a datasheet row size — see the
    module-level caveat above (DRAMScope / X-ray citations).
    """
    byte_start, n_bits = _region_bounds(region)
    byte_len = n_bits // 8
    if row_bytes <= 0:
        raise ValueError(f"row_bytes must be >0, got {row_bytes}")
    if n_rows <= 0:
        raise ValueError(f"n_rows must be >0, got {n_rows}")
    rng = np.random.default_rng(seed)
    anchor_byte = byte_start + int(rng.integers(0, byte_len))
    first_row_start = (anchor_byte // row_bytes) * row_bytes
    # Valid col_offsets are those whose hit byte in the first row still lies inside `region`;
    # this range is never empty because anchor_byte - first_row_start is always a member.
    region_hi = byte_start + byte_len
    lo_valid = max(0, byte_start - first_row_start)
    hi_valid = min(row_bytes, region_hi - first_row_start)
    col_offset = int(rng.integers(lo_valid, hi_valid))
    col_bit = int(rng.integers(0, 8))
    positions = []
    for i in range(int(n_rows)):
        row_start = first_row_start + i * row_bytes
        byte_pos = row_start + col_offset
        if byte_pos >= len(buf):
            break                                     # stripe clamped at buffer end
        positions.append((byte_pos, col_bit))
    flip_bits(buf, positions)
    coverage_hi = min(first_row_start + int(n_rows) * row_bytes, len(buf))
    record = {
        "shape": "device_column",
        "anchor_byte": int(anchor_byte),
        "coverage": [int(first_row_start), int(coverage_hi)],
        "bits_flipped": len(positions),
        "row_bytes": int(row_bytes),
        "col_offset": int(col_offset),
        "col_bit": int(col_bit),
        "n_rows": int(n_rows),
    }
    return positions, record


# --- model 3: temporally clustered burst flip --------------------------------

def temporal_burst(buf, region, timeline, seed):
    """Apply the flips scheduled for the trigger tick(s) of a quiet→burst ``timeline``.

    ``timeline`` is a list of per-tick flip counts, e.g. ``[0, 0, 0, 12, 0]`` = three quiet
    ticks then a 12-flip burst. This one-shot form applies the SUM of the timeline as a
    single burst of randomly-placed flips in the region and returns their positions. For the
    cross-query accumulation the Stage-1 scrub study needs, use ``CumulativeCorruption``,
    which steps through the timeline tick by tick and keeps state. Deterministic in ``seed``.
    """
    byte_start, n_bits = _region_bounds(region)
    total = int(sum(int(t) for t in timeline))
    if total < 0:
        raise ValueError(f"timeline has negative flip count: {timeline!r}")
    if total > n_bits:
        raise ValueError(f"burst of {total} flips exceeds region ({n_bits} bits)")
    rng = np.random.default_rng(seed)
    bit_offsets = rng.choice(n_bits, size=total, replace=False)
    positions = [_bit_to_pos(byte_start, int(b)) for b in bit_offsets]
    flip_bits(buf, positions)
    return positions


class CumulativeCorruption:
    """Stateful temporal-burst injector: corruption accumulates across ticks.

    Drives the Stage-1 scrub-interval experiment, where damage builds up over a sequence of
    queries until a scrub repairs it. ``step(tick_flips)`` adds ``tick_flips`` new random
    flips to the buffer (on top of everything already flipped, never re-selecting an
    already-flipped bit), so the corrupted-bit count grows monotonically — quiet ticks add 0,
    a trigger tick adds a burst. ``cumulative_curve`` returns the running corrupted-bit count
    per tick (the quiet→step signature the verification checks). ``reset`` restores the buffer
    and clears state, modelling a scrub. Deterministic in ``seed``.
    """

    def __init__(self, buf, region, seed):
        self.buf = buf
        self.byte_start, self.n_bits = _region_bounds(region)
        self.rng = np.random.default_rng(seed)
        self._flipped_bits = set()        # bit offsets (region-relative) currently flipped
        self._curve = []                  # cumulative corrupted-bit count after each step

    def step(self, tick_flips):
        """Add ``tick_flips`` new flips (0 on a quiet tick). Returns positions added this tick."""
        tick_flips = int(tick_flips)
        if tick_flips < 0:
            raise ValueError(f"tick_flips must be >=0, got {tick_flips}")
        available = self.n_bits - len(self._flipped_bits)
        if tick_flips > available:
            raise ValueError(f"tick wants {tick_flips} flips, only {available} bits left")
        added = []
        chosen = 0
        # Rejection-sample fresh bits so cumulative corruption never double-counts a bit.
        while chosen < tick_flips:
            b = int(self.rng.integers(0, self.n_bits))
            if b in self._flipped_bits:
                continue
            self._flipped_bits.add(b)
            added.append(_bit_to_pos(self.byte_start, b))
            chosen += 1
        flip_bits(self.buf, added)
        self._curve.append(len(self._flipped_bits))
        return added

    def run(self, timeline):
        """Step through a whole timeline (list of per-tick flip counts). Returns the curve."""
        for tick_flips in timeline:
            self.step(tick_flips)
        return self.cumulative_curve()

    def cumulative_curve(self):
        """Running corrupted-bit count after each step — quiet ticks plateau, bursts step up."""
        return list(self._curve)

    @property
    def corrupted_fraction(self):
        """Fraction of region bits currently corrupted."""
        return len(self._flipped_bits) / self.n_bits

    def reset(self):
        """Undo all accumulated flips (a scrub) and clear state."""
        positions = [_bit_to_pos(self.byte_start, b) for b in self._flipped_bits]
        flip_bits(self.buf, positions)
        self._flipped_bits.clear()
        self._curve.clear()
