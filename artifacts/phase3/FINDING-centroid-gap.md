# FINDING: the centroid gap is two problems, and the paper attributes both to the wrong cause

Status: implemented and measured. Code + experiments only — no `paper/main.tex` edits in this
pass. Run 2026-08-03 on the x86 workstation against `hnsw_M16_efC200_b7.index` (280,573,456 B).

## TL;DR

The paper says the centroids are unprotected because "they sit in an index header this
implementation does not protect." **Both halves of that are wrong.** The centroids are their own
8 KB block at offset 156; the header is a separate 156 B block at offset 0. Both were already
tagged `critical_global` in the region map — nothing consumed the tag at runtime.

The centroid row of Table 1 also pools **two independent failures**: genuine centroid damage
(silent wrong) and header destruction (crash), the latter because an 8 KB device row is aligned
to the buffer start and takes `[0, 8192)` — the header with it.

One mechanism fixes both. The cliff layer already did replicate → majority-vote → CRC → reload;
it was just hard-coded to the rotation. It now takes a region list
(`--cliff-regions rotation,centroids,header`). No new arm, no new counter, no change to
`classify_outcome`.

| centroids, 120 injections | Cr | Si | De | Re | To |
|---|---|---|---|---|---|
| unprotected | 48 | 23 | 0 | 0 | 49 |
| protected, before | 29 | 20 | 5 | 0 | 66 |
| **protected, after** | **0** | **0** | 1 | 89 | 30 |

Rent 5.7027% → **5.7086%** (the paper's `5.70\%` is unchanged). 427 tests pass. The rotation
column is byte-identical to the archived baseline, so the vectorized vote is neutral.

**Two things a reader should not skip:** the scrub costs **17.3 µs** per call for the three
regions, which is 1.2% of a search at ef=2000 but **38.5% at ef=64**, so §3.1's "disappears next
to one graph traversal" needs a depth qualifier — and 81% of that is Python dispatch, not byte
work, so the number to quote depends on which question is being asked (§3 below). The 8 KB is
also `num_cluster × padded_dim × 4` with SIFT1M using only 16 clusters, so it grows with the
corpus (§5 below).

Which paper claims must change is listed in §7.

## 1. The claim that was wrong

The paper states three times that the centroids are unprotected because "they sit in an index
header this implementation does not protect."

They do not. `qp/rabitq/layout.py:196-235` puts the centroids in their **own 8,192-byte block at
offset 156**, already a first-class region tagged `protect="critical_global"` and already listed
in `GLOBAL_FIELDS`. The header is a separate 156-byte block at offset 0. Nothing consumed the
`protect` tag at runtime, which is the actual reason neither was covered.

The centroid row of Table 1 pools two independent failures:

| fault mode | protected outcome, before | actual cause |
|---|---|---|
| `single_cell` | 7/30 `silent_wrong` | 1 byte of genuine centroid data, ΔRecall 0.056 |
| `miscorrection_word` | 13/30 `silent_wrong` | 3 bytes of genuine centroid data |
| `device_row` | **29/30 `crash`** | the 8 KB row is aligned to the buffer start (`qp/faults.py:208`), so it covers `[0, 8192)` and destroys the header; `load()` then mallocs on garbage geometry (`hnsw.hpp:810-818`) |

**Measured proof of the split.** Protecting the centroids *alone* leaves `device_row × centroids`
at 29/30 crashes — bit-for-bit the previous result. The crash is the header, not the centroids.

## 2. What was done

One mechanism, not two. The cliff layer already replicated, majority-voted, CRC-verified and
reloaded-from-disk a global region; it was just hard-coded to the rotation
(`phase3_e5_recovery.py:104`). It now takes a list, via `--cliff-regions`:

- **centroids** — same structural class as the rotation: global, read by every query
  (`q_to_centroids`), small enough to replicate.
- **header** — a different argument. `load()` parses it into scalars once and never re-reads it,
  so the per-query justification does not apply. But the scrub runs immediately before
  `deserialize_index`, which is exactly when a load-time check wants to run.

No new arms (`ARMS` is threaded through `classify_outcome` and the row-count gates), no new
counter, no change to `classify_outcome`. The ablation is two values of one flag.

The per-bit triple loop in `_scrub_cliff` had to be vectorized first: `bl*8*R` went from 1,536
iterations to 196,608 on a per-query path. `(a&b)|(b&c)|(a&c)` is exact for R=3; larger R falls
back to `unpackbits`. Verified bit-identical to the old loop (majority **and** `repaired_bits`)
across R ∈ {1..5}. What it costs is §3.1.

## 3. Results

### Centroid column, pooled over all four fault modes (120 injections)

| | Cr | Si | De | Re | To |
|---|---|---|---|---|---|
| unprotected | 48 | 23 | 0 | 0 | 49 |
| protected, **before** (rotation only) | 29 | 20 | 5 | 0 | 66 |
| protected, **after** (+ centroids + header) | **0** | **0** | 1 | 89 | 30 |

The unprotected row reproduces the archived baseline exactly (48/23/49), which is the control
that says the fault injection itself did not move.

### Per cell (30 seeds, arm ON)

| shape | rotation+centroids | rotation+centroids+header |
|---|---|---|
| `single_cell` | 30 repaired | 30 repaired |
| `device_row` | **29 crash**, 1 detected | **0 crash**, 29 repaired, 1 detected |
| `device_column` | 30 tolerated | 30 tolerated |
| `miscorrection_word` (E9, ef=64) | — | 30 repaired |

`device_column` holds recall in all 30 and the cliff layer does act in every one, but the stripe
also smears into the element payload, so the ex-data CRC still flags elements and
`classify_outcome` calls that `tolerated` rather than `repaired`. It is not a case of the layer
doing nothing.

### Falsification checks (all four from the plan)

- **repaired-but-recall-did-not-return: 0 of 174.** No damage escaped the protected window.
- **rotation regression: all 3 shapes match the archived baseline exactly.** The vectorized vote
  is neutral.
- **`device_row × centroids` after the header guard: 0 crashes** (was 29).
- **per-query cost: see §3.1.** It is the one check that did not come out clean, and the CRC-only
  variant should stay on the table rather than be dismissed.

Attribution over the 300 protected evals of the full configuration: centroids acted in 120
trials (932,919 bits), header in 30 (18,144 bits), rotation in 30 (90 bits).

### 3.1 Per-query cost — and a correction

An earlier revision of this document quoted two different numbers for this and computed its
percentages from a third. All three are recorded here because the difference between them is the
whole point:

| number | what it actually measures |
|---|---|
| 8.3 µs | `_majority_vote` for the **centroids region alone** — one of three, not the whole scrub |
| 3.27 µs | a **constructed** figure: the bare `(a&b)\|(b&c)\|(a&c)` timed in isolation, plus `zlib.crc32`. No code path does only that — `_majority_vote` also runs 4 × (XOR + `.any()`) |
| **17.3 µs** | **`_scrub_cliff` end to end for all three protected regions.** This is the cost. |

`bench_cliff_cost.py` now produces all of it in one run and gates that the parts add up to the
whole, so the number cannot drift from its own decomposition again. Figures below are one run
(`artifacts/phase3/e10/cliff_cost.json`); they move ~3% run to run.

| component | µs | share |
|---|---|---|
| 3 × 4.69 numpy dispatch — **size-independent** | 14.07 | 81% |
| 8,412 B × 0.199 µs/KB — the actual byte work | 1.63 | 9% |
| `zlib.crc32` over the same bytes | 1.25 | 7% |
| predicted / measured | 16.95 / **17.34** | reconciles to 2.2% |

Slope and intercept come from fitting `_majority_vote` over 64 B … 64 KB:
`cost = 4.69 µs + 0.199 µs/KB`.

Against one search (search-only; load separated out by fitting wall across ef):

| | ef=2000 (1,405.91 µs/query) | ef=64 (44.99 µs/query) |
|---|---|---|
| measured, as implemented | **1.23%** | **38.5%** |
| size-dependent floor (1.63 + 1.25 = 2.88 µs) | 0.21% | 6.4% |

**Which row to quote depends on the question.** The floor is what survives a C++/SIMD
implementation; the measured row is what this harness does. Neither justifies the paper's current
"comparing it disappears next to one graph traversal" without a depth qualifier: even at the
floor it is 6.4% of a search at ef=64.

A second earlier claim, that "cost scales with region count, not size", is also **wrong**. It
rested on the coincidence `3 × 5.81 ≈ 17.53`. The slope fit shows a real size term of
0.199 µs/KB. Dispatch overhead dominates it at 8.4 KB, but the size term is exactly what grows
with `num_cluster` (§5), so it must not be dismissed.

Caveat on all of the above: in E6/E9 the scrub runs once per *eval* — once per 10,000-query
batch — not per query, so none of these numbers affect the outcome tables. The per-query framing
is §3.2's design intent, not what the sweep measured.

### Cost

`phase3_cost.mem_cost` over `{rotation, centroids, header} × {mult:3, checksum:4}`:
16,836 B of replication + the 16,000,064 B CRC manifest on a 280,573,456 B index.

**Rent 5.7027% → 5.7086%.** The paper's `5.70\%` is unchanged at the precision it is quoted.

## 4. The header predicate, and an honest limit

The harness repairs the header from the durable original. A loader cannot: the file it is reading
**is** the corrupted one. So `header-guard.patch` detects and throws, exactly like the framing
guard — the plan's expectation that it could "escalate to repair by pread" does not hold on that
path.

`phase3_e10_header.py` enumerates all 1,248 single-bit header flips (not sampled):

| tier | detected |
|---|---|
| header-only — what C++ `load()` can afford | 1,129 / 1,248 = **90.46%** |
| + walking the record stream to EOF | 1,172 / 1,248 = **93.91%** |

The 119 header-only escapes, classified by what they would actually do:

- **19 desync** — move a computed offset, so the sequential read misaligns.
- **24 level_overrun** — send the descent to a level that was never allocated. 4 by raising
  `maxlevel`; **20 by moving `enterpoint_node` to a different *valid* element id**. That one
  looks harmless and is not: exactly one element in this index carries a 340-byte record
  (histogram `{0: 937001, 68: 59126, 136: 3626, 204: 234, 272: 12, 340: 1}`), so any other id
  reads past the end of a shorter link list. Replayed against the **stock** binary: SIGSEGV.
- **25 alloc + 51 benign** — values a legitimate index could hold and that `load()` acts on
  identically. Rejecting these would mean rejecting legitimate indexes, which is the bar
  `framing-guard.patch` set for itself.

All 43 dangerous ones close under the full walk, which pins three things a header-only check
cannot: the stream must end exactly where the rotation begins, the longest record must equal
`maxlevel × size_links_per_element`, and the entry point's own record must be that longest one.
**The header guard and the shipped framing guard are complementary, not alternatives** — verified,
not assumed.

### Rebuild verification

Scratch build; production `bin/` untouched. ef=64, k=10, median of 10.

| | stock | guarded |
|---|---|---|
| clean recall@10 | 0.95035 | 0.95035 |
| clean top-10 ids | `sha256[:16] fbe27be5c4bd6264` | **identical** |
| load+search wall | 0.7129 s | 0.7111 s |

The difference is −0.0018 s (−0.25%) — the guarded binary came out nominally *faster*. That is
noise: the stock binary's own run-to-run spread is 0.0258 s, fourteen times the difference. The
honest statement is that the cost is **below the measurement floor**, which is what ~20 integer
comparisons executed once per load should look like, against the framing guard's +0.75% (4
comparisons × 1,000,000 elements).

Replaying one high-bit flip into each of the 18 load-bearing fields: 13 rejected by the guard with
a diagnostic, 1 rejected earlier by existing code (`ex_bits` via `select_excode_ipfunc`), 3 crash
identically to stock (`num_cluster`, `maxlevel`, `enterpoint_node` — exactly the classes the
coverage analysis predicts a header-only check cannot close), 1 loads normally (`max_elements`
raised, a legitimate value). Four control flips outside the header do not trigger the guard.

## 5. Caveat to carry into the paper pass

The 8 KB centroid figure is `num_cluster × padded_dim × 4`, and SIFT1M uses only **16 clusters**.
On a larger corpus the block grows and the "replicate it, it is nearly free" argument weakens.
It weakens in both directions, and §3.1 now quantifies one of them: memory grows linearly, and so
does the vote's size-dependent term at 0.199 µs/KB. At 16 clusters that term is 9% of the scrub
and easy to overlook; at 1,024 clusters the centroid block is 512 KB and the term alone is
**101.8 µs** (measured extrapolation, `bench_cliff_cost.py`), which no search depth hides.
§3.2's cost sentence should say what it scales with. Protecting the centroids puts this in front
of a reviewer whether or not the work is done.

## 6. Reproduce

```bash
cd /home/u01/tmouse/quant_protection && source .venv/bin/activate
python -m pytest artifacts/phase3/tests/ -q     # 427 passed
./run_centroid_sweeps.sh                        # 10 shards, ~30 min wall
python analyze_centroid_sweeps.py               # the tables above
python phase3_e10_header.py                     # the 1,248-case enumeration
python bench_cliff_cost.py                      # per-query cost + its decomposition
```

Artifacts: `artifacts/phase3/e6/centroid_gap_summary.json`,
`artifacts/phase3/e10/header_coverage.json`, `artifacts/phase3/e10/header_guard_rebuild.json`,
`rabitq_instrumentation/header-guard.patch`.

Note: `artifacts/phase3/e*/raw/` is gitignored by existing convention, so the per-trial
`*.records.jsonl` are not in the repo. `centroid_gap_summary.json` carries every number the
falsification checks produced; re-run the sweep if you need the per-trial `field_hits`.

## 7. What this means for `paper/main.tex`

**Must change — currently factually wrong**

1. §3.1 `\head{Centroids uncovered}` — the whole paragraph. Both the factual claim (they are not
   in the header) and the status (now covered). It becomes the second instance of the cliff rule
   rather than its exception.
2. Table 1 `tab:region-outcomes`, centroids row, protected half: `29 20 5 0 66` → `0 0 1 89 30`.
   The unprotected half stays — we reproduced 48/23/49 exactly.
3. §4.1 `\head{What protection converts}` — "in every region but one" → "in every region"; delete
   the centroid exception sentence.
4. §4.1 `\head{Small is not benign}` — "zero crashes and 13 silent failures, all in the centroids"
   → zero crashes and zero silent failures. (The unprotected 17/43 are unchanged and verified.)
5. §5 Conclusion — "The one exception is the centroids…" → the remaining honest gap is
   `upper_links`, which §4.1 already names.

**Must update — not wrong, but stale**

6. "comparing 64 bytes three ways" (§3.1) → 8,412 bytes.
7. "the manifest plus the replicated rotation" → "…the replicated shared parameters" (§3.2, §4.3).
8. "Replicate the 64-byte cliff" (§5) and "The 64 bytes on the per-query critical path" (§3) →
   8.4 KB.
9. The rent figure itself does **not** change.

**Can now be strengthened**

10. §5 "in every region **they reach**" — drop the qualifier; it was there to dodge the centroids.
11. §3 `\head{Protecting the locating bytes}` / §4.1 `\head{Guarding the parse}` — the header is a
    second instance of the same principle, and the complementarity with the framing guard is
    measured (§4 above), not asserted.

**Must add as caveats**

12. The 8 KB scales with `num_cluster × padded_dim` (§5 above).
13. The per-query cost holds at depth (0.21% of a search at ef=2000) but not at ef=64, where
    even the byte-work floor is 6.4% (§3.1 above). §3.1's "comparing it disappears next to one
    graph traversal" needs a depth qualifier, and the paper should say which quantity it means.
14. The C++ guard detects only; repair needs a pristine source the loader does not have.

**Pre-existing tension worth resolving while editing:** §3's budget list says "the 1.4\% framing
gets nothing", but §3 `\head{Protecting the locating bytes}` and §4.1 `\head{Guarding the parse}`
both describe a framing check. If "nothing" means "no *memory* budget", say so — adding the header
guard makes the ambiguity more visible.
