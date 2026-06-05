# Phase 1 — Region Accounting

How much of each index's memory is even recall-relevant? A uniformly random physical bit flip lands in proportion to bucket size, so this fraction sets a floor on bit-flip sensitivity before quantization is even considered.

Buckets: **recall_relevant** (flips deserialize and can silently degrade recall), **crash_structure** (flips crash/are detected on load), **benign** (padding/unaccounted).


## Per-index bucket breakdown

| index | total MB | recall_relevant | crash_structure | benign |
|---|---:|---:|---:|---:|
| FLAT | 512.0 | 512.0 MB (100.0%) | 0.0 MB (0.0%) | 0.0 MB (0.0%) |
| IVF_FLAT | 520.5 | 520.5 MB (100.0%) | 0.0 MB (0.0%) | 0.0 MB (0.0%) |
| IVF_SQ8 | 136.5 | 136.5 MB (100.0%) | 0.0 MB (0.0%) | 0.0 MB (0.0%) |
| IVF_PQ_M8 | 16.7 | 16.7 MB (100.0%) | 0.0 MB (0.0%) | 0.0 MB (0.0%) |
| IVF_PQ_M16 | 24.7 | 24.7 MB (100.0%) | 0.0 MB (0.0%) | 0.0 MB (0.0%) |
| HNSW | 784.1 | 512.0 MB (65.3%) | 272.1 MB (34.7%) | 0.0 MB (0.0%) |
| HNSW_SQ8 | 400.1 | 128.0 MB (32.0%) | 272.1 MB (68.0%) | 0.0 MB (0.0%) |

## Small but high-leverage regions

Tiny regions shared by/ pointing at many vectors — a single flip here can move every dependent vector.

| index | region | bytes | bits | % of total |
|---|---|---:|---:|---:|
| IVF_FLAT | centroid | 524,288 | 4,194,304 | 0.1007% |
| IVF_SQ8 | centroid | 524,288 | 4,194,304 | 0.3840% |
| IVF_SQ8 | sq_scale | 1,024 | 8,192 | 0.0007% |
| IVF_PQ_M8 | pq_codebook | 131,072 | 1,048,576 | 0.7866% |
| IVF_PQ_M16 | pq_codebook | 131,072 | 1,048,576 | 0.5314% |
| HNSW_SQ8 | sq_scale | 1,024 | 8,192 | 0.0003% |

## Headline prediction (A4)

**IVF_SQ8 vs HNSW_SQ8 — same scalar quantizer, opposite memory profile.** IVF_SQ8 is **100.0%** recall-relevant (nearly its whole footprint is codes), so almost every random flip hits recall. HNSW_SQ8 is only **32.0%** recall-relevant — about 68% of it is graph structure (crash-detectable), so at the same faults/MB most flips land on detectable structure rather than silently corrupting recall. Predicted recall-relevant gap: **68.0 percentage points**. The single-bit sweep (Task B) tests whether this footprint difference, not the quantizer, dominates sensitivity.

