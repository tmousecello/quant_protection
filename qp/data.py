"""SIFT1M dataset loaders.

`read_fvecs` is the reference loader factored out of verify_env.py: each row is an
int32 dim count followed by `dim` float32 values; we reshape to (N, dim+1) and drop the
leading dim column. `read_ivecs` is the int32 analogue for the ground-truth file.

Accessors cache by path so repeated calls in a phase are free. `max_rows` lets smoke
runs load only a subset of the 1M base.
"""
import numpy as np

from . import config

_CACHE = {}


def read_fvecs(path, max_rows=None):
    """Read .fvecs -> (N, dim) float32. Reads only `max_rows` rows if given."""
    with open(path, "rb") as f:
        dim = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        f.seek(0)
        row_floats = dim + 1
        count = max_rows * row_floats if max_rows is not None else -1
        raw = np.fromfile(f, dtype=np.float32, count=count)
    raw = raw.reshape(-1, dim + 1)
    return np.ascontiguousarray(raw[:, 1:]), dim


def read_ivecs(path, max_rows=None):
    """Read .ivecs -> (N, dim) int32. Used for ground-truth neighbor ids."""
    with open(path, "rb") as f:
        dim = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        f.seek(0)
        row_ints = dim + 1
        count = max_rows * row_ints if max_rows is not None else -1
        raw = np.fromfile(f, dtype=np.int32, count=count)
    raw = raw.reshape(-1, dim + 1)
    return np.ascontiguousarray(raw[:, 1:])


def _cached(key, loader):
    if key not in _CACHE:
        _CACHE[key] = loader()
    return _CACHE[key]


def load_base(max_rows=None):
    """1M x 128 float32 base vectors (or first `max_rows`)."""
    xb, _ = _cached(("base", max_rows), lambda: read_fvecs(config.SIFT_BASE, max_rows))
    return xb


def load_query(max_rows=None):
    """10k x 128 float32 query vectors."""
    xq, _ = _cached(("query", max_rows), lambda: read_fvecs(config.SIFT_QUERY, max_rows))
    return xq


def load_groundtruth(max_rows=None):
    """10k x 100 int32 true-neighbor ids for the full 1M base."""
    return _cached(("gt", max_rows), lambda: read_ivecs(config.SIFT_GT, max_rows))


def load_learn(max_rows=None):
    """100k x 128 float32 training vectors (disjoint from base)."""
    xl, _ = _cached(("learn", max_rows), lambda: read_fvecs(config.SIFT_LEARN, max_rows))
    return xl


def clear_cache():
    _CACHE.clear()
