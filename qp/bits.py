"""Pure-numpy bit primitives — the faiss-free core of the corruption substrate.

These were extracted from qp/flip.py so modules that only need byte/bit manipulation
(e.g. qp.faults, the RaBitQ adapter) can import them WITHOUT pulling in faiss. flip.py
re-exports every name here, so existing callers (`from qp.flip import flip_bit`, etc.)
keep working unchanged.

Bit addressing convention (shared by everything downstream): an absolute bit index p
maps to byte ``p // 8``, bit ``p % 8``. ``buf`` is always a writable uint8 numpy array.
Every flip here is a self-inverse XOR, so applying the same positions twice restores the
buffer exactly — this is what makes inject -> restore roundtrips byte-identical.
"""
import numpy as np


def flip_bit(buf, byte_pos, bit):
    """In-place: flip `bit` (0..7) of byte `byte_pos`."""
    buf[byte_pos] ^= np.uint8(1 << bit)


def flip_bits(buf, positions):
    """In-place: flip many (byte_pos, bit) pairs. Used by the multi-bit rate sweep."""
    for byte_pos, bit in positions:
        buf[byte_pos] ^= np.uint8(1 << bit)


def burst_positions(start_bit, B):
    """The (byte_pos, bit) pairs for B consecutive bits starting at absolute bit `start_bit`.

    Bit addressing matches flip_bit: absolute bit p -> byte p // 8, bit p % 8. Returned as a
    plain list so the isolated workers (qp.isolation) can pickle the positions and apply them
    with flip_bits without re-deriving the arithmetic.
    """
    return [(p // 8, p % 8) for p in range(int(start_bit), int(start_bit) + int(B))]


def burst_flip(buf, start_bit, B):
    """In-place: flip B consecutive bits beginning at absolute bit index `start_bit`.

    Models a spatially-clustered fault (a bad DRAM block / contiguous storage corruption),
    the burst sub-study's injector. Self-inverse like flip_bit — calling it again with the
    same (start_bit, B) restores the buffer exactly. The single-bit interface is untouched.
    """
    flip_bits(buf, burst_positions(start_bit, B))
