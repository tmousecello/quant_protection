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
