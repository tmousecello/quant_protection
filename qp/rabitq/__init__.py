"""RaBitQ adapter for Phase 3 — byte-region map + subprocess wrappers around Samuel's binaries.

This subpackage is the ONLY RaBitQ-specific code in qp/. It does not re-implement any
metric or fault logic (those stay in qp.metrics / qp.faults); it only:
  - layout.py  : the serialized-index byte-region map (header / centroids / per-element
                 links|cluster_id|label|bin|ex / rotation), derived from the RaBitQ-Library
                 source at the pinned commit (see build_rabitq.sh). Resolves the rotation
                 region and on-disk layout from source — no guessing.
  - adapter.py : serialize(read)/deserialize(write) of the index file, dataset/query/gt
                 loaders, and subprocess wrappers (build / query -> neighbour ids) that the
                 recall path feeds into qp.metrics. All live calls gate on binaries_built().

Live measurement (clean baseline ~0.983, parity) requires the C++ binaries, which build
ONLY on x86-64 (the library is AVX2/AVX512-only). On Apple Silicon the layout is still fully
readable/derivable from source; the binaries are produced by running build_rabitq.sh on an
x86-64 Linux host.
"""
