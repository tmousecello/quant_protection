# Claim → implementation → evidence-level map

Every mechanism claim made in §3 of `paper_draft_zh.md`, the code that actually implements it,
and how strongly it is supported. Written as the base for the third audit round and as the
wording authority when §3 is drafted: **do not describe a mechanism in the paper in stronger
terms than its row here.**

Evidence levels, following `artifacts/phase3/audits/STAGE3_AUDIT_ROUND2.md`:

| level | meaning |
|---|---|
| **measured** | A number came out of a run that could have come out otherwise. |
| **by construction** | The code makes it true; the check verifies self-consistency, not a fact about the world. Real, but never phrase it as a finding. |
| **design claim, unmeasured** | Implemented and reasoned about, but nothing measures it. Say so in the paper. |
| **not implemented** | Described in §3, no code behind it. |

C++ line numbers are for the patched header
`third_party/RaBitQ-Library/include/rabitqlib/index/hnsw/hnsw.hpp` as produced by
`rabitq_instrumentation/recovery-changes.patch` (regenerate with `bash build_rabitq.sh`).

---

## §3.1 / §3.3 — slope layer: on-access detection, EB degradation, counting, batch reload

| # | Claim (§) | Implementation | Level |
|---|---|---|---|
| 1 | **On-access detection** — "重排序取用某候選的 ex 區塊前驗其 CRC" (§3.1) | `hnsw.hpp:164` `ex_corrupted_now()`, dispatched at `:202` `ex_bad()`, armed by `load_crc_manifest(..., lazy=true)` `:236`; selected by `exp_dumpids --crc-mode lazy` | **measured** — `expb_lazy_gates.json`, all five gates green (see *Status* below). Gate 2: load and lazy return bit-identical top-k ids and identical recall under both policies. |
| 2 | **Piggyback — no extra memory traffic** (§3.1 「搭便車」, §3.4 「零額外計算」) | EB path only: `hnsw.hpp:1571-1573` computes the verdict immediately before `get_full_est()` reads the very same ex window | **measured — and the claim is FALSE for `drop`.** EB **+7.74%** of search wall (112 µs/query, 771 checks); drop **+224.24%** (3.27 ms/query, 20 634 checks) — drop more than triples query latency. See "The piggyback claim is policy-specific" below. This is the row that most needs the paper's wording fixed. |
| 3 | **Detection emits the decision variable f̂ for free** (§3.3 「偵測直接吐出決策變數」) | `phase3_g_detection.py` `observed_load = elements_crc_fail / elements_checked`; numerator pinned by `phase3_expb_recovery._check_crc_fail_count`, denominator by `extract_points`' own check | **by construction** (unchanged from round 2 — `g_detection_summary.json` already carries the honest note). Note this holds for the *load* scan; `extract_points` now rejects lazy rows, whose ratio is the different, access-weighted quantity. |
| 4 | **EB degradation beats naive 1-bit fallback** (§3.3) | `hnsw.hpp:1591` `rank_dist = est + (est − low)`, Samuel exp-3 semantics, unchanged | **measured** — expb sweep, `expb_summary.json` (`eb_minus_drop` / `eb_minus_none`), gate 3 anchored to Samuel's F3 median 0.941865 |
| 5 | **Detection raises tolerable corruption ~29×** (§3.3, abstract) | `phase3_f_synth.py` `tolerance_curves` / `solve_fstar` over the expb light sweep | **measured** (interpolated from the measured curve) |
| 6 | **Counter → f̂ drives scheduling** (§3.1 「累加損壞計數」) | `FiQueryStats.corrupt_hits` / `consults` (`hnsw.hpp:148`), surfaced as `stats.totals`; consumed by `phase3_e5_recovery.scrub_if_due` | **by construction** |
| 7 | **Batch reload of damaged chunks** (§3.1, §3.3 「群組 I/O 批次重載」) | `phase3_e7_episode.py:312` `batch_repair`, `:288` `reload_bytes`; threshold edge-trigger at `:262` | **measured** (repair_io_bytes / repair_wall_s are measured; the "one seek amortised over N" argument itself is analytic) |
| 8 | **Corruption is independent of access, so the consulted-set ratio is unbiased** (§3.3) | — | **design claim, unmeasured.** The draft already scopes it out for adversarial rowhammer; it is not tested for retention faults either. |

## §3.2 — cliff layer: replica majority vote and self-scrub

| # | Claim | Implementation | Level |
|---|---|---|---|
| 9 | **Per-query majority-vote repair of the rotation** | `phase3_e5_recovery.py:196`, `:258` `_scrub_cliff` | **measured** — 200-tick timeline, `cliff_repaired=2012`, recall flat at 0.98376 |
| 10 | **Never overwrite with a wrong majority** | `phase3_e5_recovery.py:286` — majority CRC checked against `_rot_clean_crc` before writing | **by construction** |
| 11 | **Un-scrubbed replicas are a one-shot fuse** | 30-seed first-failure distribution | **measured** (median 21, IQR [9,30], N=30 — the most solid result in the study) |
| 12 | **Event-triggered full reload makes protection renewable** | `_cliff_vote_fail_reloads` / `_cliff_reload_triggered` | **measured within one 200-tick run**; "無限" remains extrapolation (round-2 finding, unchanged) |

## §3.4 — bounds-check layer

| # | Claim | Implementation | Level |
|---|---|---|---|
| 13 | **Bounds-check skip turns pointer corruption into a skip, not a crash** | `hnsw.hpp:1549` `fi_safe_mode_` guard | **by construction**; `oob_restored` is structurally ~0 on a 1M index and `phase3_e6_shapes.py:573` refuses to report fictitious counts for exactly this reason |

---

## The piggyback claim is policy-specific (new, and it constrains the wording)

The two detecting policies consult the predicate at different points, and only one of them is
free:

| policy | consult site | when it fires | are the CRC'd bytes read anyway? | measured cost |
|---|---|---|---|---|
| `fallback_eb` | `hnsw.hpp:1571-1573` | only for candidates that reach rerank (`flag_update_KNNs`) | **yes** — `get_full_est()` loads the same window on the next line. Genuine piggyback. | **+7.74%** of search wall, 771 checks / 74 KB per query |
| `drop` | `hnsw.hpp:1539` | **every unvisited neighbour**, before the bounds guard and before `get_bin_est` | **no** — a dropped node's ex block is never read. The CRC is extra memory traffic. | **+224.24%** of search wall, 20 634 checks / 1.98 MB per query |

`crc_bytes(drop) ≫ crc_bytes(fallback_eb)` was the prediction and the measurement confirms it:
**26.8× the bytes, 29.2× the time.** The time ratio tracking the byte ratio so closely is itself
the evidence that the cost is CRC-input-bound rather than an artefact of either policy's control
flow. "偵測不增加記憶體流量" is defensible for the EB policy only; for `drop` the detection layer
costs more than the search it protects.

`--lazy-gate` measures both on purpose.

Moving `drop`'s consult to the access point would fix this, but it would change what
`FAULT_DROP` means (a dropped node currently contributes neither result nor routing) and
invalidate every frozen expb record. That is option B, ruled out this round.

## Wording fixes for §3 (suggestions for the writer; the draft is not edited here)

1. §3.4 「零額外計算」 → 「無額外**記憶體流量**(CRC 算在 rerank 即將載入的同一批 bytes 上;
   此點對 `fallback_eb` 成立,對 `drop` 不成立——drop 的判定在圖遍歷期、被丟棄的節點其 ex 區塊
   本就不會被讀)。**計算**開銷實測:EB **+7.74%**、drop **+224%**(ef=2000, k=10, SIFT1M,
   軟體 CRC32 ~0.79 GB/s);硬體 CRC32C 可再降一個數量級(future work)。」
   ——注意 drop 那個數字不是「開銷偏高」,而是偵測比它保護的搜尋本身還貴,措辭不可含糊。
2. §3.4 / §4 的 0.03–5.7% 是 **記憶體(manifest)** 開銷。查詢路徑的**時間**開銷是另一根軸,
   已由 `--lazy-gate` 量測,承載於 `frontier.csv` 的 `detect_pct_of_search_*` 欄位
   (EB 7.74 / drop 224.24)。兩者不可混用,也不可相加——摘要目前把「0.03–5.7%」講成
   「偵測層開銷」,那只是記憶體那一根軸,讀者會誤以為時間開銷也在該區間內。
3. §3.1 描述 on-access 時,可加一句限定:兩種時機(載入期全掃 / access 時重算)在「查詢期
   記憶體不可變」下逐位元等價,而本文的 fault model 正是不可變假設不成立的場景——這是選擇
   on-access 的理由,不只是效率考量。
4. §3.3 引用 G 時必須保留 round-2 的限定語:「構造性」而非「偵測準確率」。

## Status of row 1 — what is and is not verified

**Executed 2026-08-02 on `meow1` (AMD Ryzen 9 9950X, AVX512BW present), all five gates green.**
Artifact: `artifacts/phase3/expb/expb_lazy_gates.json` (`uniform_accum`, f=0.05, ef=2000, k=10,
seed 1234, 4 flips/element). Procedure: `artifacts/phase3/EXPB_RUNBOOK.md` §5.

| gate | result |
|---|---|
| 0 build has `--crc-mode` | PASS |
| 1 clean index flags nothing in either mode | PASS — both policies, both modes, 0 flagged |
| 2 **load vs lazy identical ids** | PASS — `np.array_equal` on all 10 000×10 ids, recall equal to the last digit (EB 0.94329, drop 0.94050) |
| 3 `0 < lazy distinct ≤ load crc_fail` | PASS — EB 47 227, drop 49 994, both ≤ 50 000 injected |
| 4 overhead measured | reported, never asserted |

Gate 2 is the load-bearing one and it is a *consequence*, not a hypothesis: the decision points
did not move, so identical ids is what "only the timing changed" predicts. Its value is as a bug
detector — a difference would have meant `ex_bad()` / `ex_corrupted_now()` disagree — so report it
in the paper as verification, never as a finding. (Per `audits/GATE_DESIGN_MEMO.md`: it is a real
gate because its fail condition is stateable, but it is not evidence about the world.)

**Which overhead number to quote.** `headline_delta_ns_per_query` — the `search_wall_ns` delta
between the two runs — is the one for §3. It carries no per-call timer, and it is meaningful
precisely *because* gate 2 proved both runs perform the same search work. The two cross-checks
agree in magnitude and, usefully, in the order theory predicts:

| policy | headline (quote this) | analytic | timed |
|---|---|---|---|
| `fallback_eb` | 111 795 ns/q (+7.74%) | 93 918 (0.84×) | 149 482 (1.34×) |
| `drop` | 3 269 525 ns/q (+224.24%) | 2 509 884 (0.77×) | 3 857 572 (1.18×) |

`analytic` (= `crc_bytes` × the load scan's ns/byte) sits *below* the headline because the load
scan is sequential over 96 MB and misses the random-access penalty the query path actually pays;
`timed` sits *above* it because each check is bracketed by two `steady_clock` reads (~40–50 ns
against a ~190 ns measured per-check cost). The headline falling between the two bounds is the
cross-check passing, not a discrepancy to explain away.

**Scope.** One host, one index, one corruption config, software CRC32 at ~0.79 GB/s measured over
the load scan. The percentages are relative to *this* search cost (ef=2000, single-threaded); a
cheaper search or a hardware CRC32C would move them. What is robust is the EB↔drop ratio, which is
set by consult-site geometry, not by CRC speed.
