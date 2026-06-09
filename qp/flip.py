"""Bit-flip corruption substrate.

The proven method (verify_env.py): faiss.serialize_index(index) -> numpy uint8 array ->
XOR a bit at a chosen byte -> faiss.deserialize_index(buf). This module packages that so
Phase 1 (single-bit map) and Phase 2 (rate sweep) reuse exactly one implementation.

A corrupted buffer can fail to deserialize or fail at search time; `rebuild` and
`safe_search` capture the exception so callers can classify it as a `crash` failure mode.
"""
import faiss
import numpy as np


def to_buffer(index):
    """Serialize an index to a writable uint8 numpy array (a private copy)."""
    return np.asarray(faiss.serialize_index(index), dtype=np.uint8).copy()


def byte_size(index):
    """Serialized length in bytes — the denominator for faults/MB normalization."""
    return int(np.asarray(faiss.serialize_index(index)).size)


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


def rebuild(buf):
    """Deserialize a (possibly corrupted) buffer.

    Returns (index, exception). On success exception is None; on failure index is None
    and exception holds the raised error (a `crash` failure mode for the corruption study).
    """
    try:
        return faiss.deserialize_index(buf), None
    except Exception as exc:  # noqa: BLE001 - any deserialize failure is a crash mode
        return None, exc


def safe_search(index, xq, k):
    """Search guarding against crashes. Returns (D, I, exception)."""
    try:
        D, I = index.search(xq, k)
        return D, I, None
    except Exception as exc:  # noqa: BLE001
        return None, None, exc
