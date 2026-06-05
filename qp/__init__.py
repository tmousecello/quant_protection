"""qp — shared infrastructure for the quant_protection bit-flip sensitivity study.

Modules:
  config   frozen controlled variables (nlist/M/efC, operating point, paths, seed)
  data     SIFT1M loaders (read_fvecs / read_ivecs + cached accessors)
  metrics  recall@{1,10,100}, tolerant recall, failure-mode classifier
  flip     serialize -> XOR byte/bit -> deserialize corruption substrate
  indexes  design-matrix index builders + clean-recall tuning sweep
  regions  byte-region map of a serialized index (codes/centroid/scale/codebook/graph)

These are reused unchanged by Phase 1 (single-bit sensitivity map) and Phase 2
(error-rate sweep). Phase 0 (phase0_build.py) is the first consumer.
"""
