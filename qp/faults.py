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
    hit = rng.random(n_bits) < p                       # iid Bernoulli(p) per bit
    bit_offsets = np.nonzero(hit)[0]
    positions = [_bit_to_pos(byte_start, b) for b in bit_offsets]
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
    byte_start, byte_len = int(region[0]), int(region[1])
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
