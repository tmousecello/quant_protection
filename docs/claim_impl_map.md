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
| 1 | **On-access detection** — "重排序取用某候選的 ex 區塊前驗其 CRC" (§3.1) | `hnsw.hpp:164` `ex_corrupted_now()`, dispatched at `:202` `ex_bad()`, armed by `load_crc_manifest(..., lazy=true)` `:236`; selected by `exp_dumpids --crc-mode lazy` | **implemented; awaiting the acceptance run** (see *Status* below) |
| 2 | **Piggyback — no extra memory traffic** (§3.1 「搭便車」, §3.4 「零額外計算」) | EB path only: `hnsw.hpp:1571` computes the verdict immediately before `get_full_est()` reads the very same ex window | **design claim, unmeasured — and FALSE for `drop`.** See "The piggyback claim is policy-specific" below. This is the row that most needs the paper's wording fixed. |
| 3 | **Detection emits the decision variable f̂ for free** (§3.3 「偵測直接吐出決策變數」) | `phase3_g_detection.py` `observed_load = elements_crc_fail / elements_checked`; numerator pinned by `phase3_expb_recovery._check_crc_fail_count`, denominator by `extract_points`' own check | **by construction** (unchanged from round 2 — `g_detection_summary.json` already carries the honest note). Note this holds for the *load* scan; `extract_points` now rejects lazy rows, whose ratio is the different, access-weighted quantity. |
| 4 | **EB degradation beats naive 1-bit fallback** (§3.3) | `hnsw.hpp:1586` `rank_dist = est + (est − low)`, Samuel exp-3 semantics, unchanged | **measured** — expb sweep, `expb_summary.json` (`eb_minus_drop` / `eb_minus_none`), gate 3 anchored to Samuel's F3 median 0.941865 |
| 5 | **Detection raises tolerable corruption ~29×** (§3.3, abstract) | `phase3_f_synth.py` `tolerance_curves` / `solve_fstar` over the expb light sweep | **measured** (interpolated from the measured curve) |
| 6 | **Counter → f̂ drives scheduling** (§3.1 「累加損壞計數」) | `FiQueryStats.corrupt_hits` / `consults` (`hnsw.hpp:122`), surfaced as `stats.totals`; consumed by `phase3_e5_recovery.scrub_if_due` | **by construction** |
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

| policy | consult site | when it fires | are the CRC'd bytes read anyway? |
|---|---|---|---|
| `fallback_eb` | `hnsw.hpp:1571` | only for candidates that reach rerank (`flag_update_KNNs`) | **yes** — `get_full_est()` loads the same window on the next line. Genuine piggyback. |
| `drop` | `hnsw.hpp:1537` | **every unvisited neighbour**, before the bounds guard and before `get_bin_est` | **no** — a dropped node's ex block is never read. The CRC is extra memory traffic. |

So `crc_bytes(drop) ≫ crc_bytes(fallback_eb)` is expected, and "偵測不增加記憶體流量" is only
defensible for the EB policy. `--lazy-gate` measures both on purpose.

Moving `drop`'s consult to the access point would fix this, but it would change what
`FAULT_DROP` means (a dropped node currently contributes neither result nor routing) and
invalidate every frozen expb record. That is option B, ruled out this round.

## Wording fixes for §3 (suggestions for the writer; the draft is not edited here)

1. §3.4 「零額外計算」 → 「無額外**記憶體流量**(CRC 算在 rerank 即將載入的同一批 bytes 上;
   此點對 `fallback_eb` 成立,對 `drop` 不成立——drop 的判定在圖遍歷期、被丟棄的節點其 ex 區塊
   本就不會被讀)。**計算**開銷實測 X%/query;硬體 CRC32C 可再降一個數量級(future work)。」
2. §3.4 / §4 的 0.03–5.7% 是 **記憶體(manifest)** 開銷。查詢路徑的**時間**開銷是另一根軸,
   由 `--lazy-gate` 量測、以 `frontier.csv` 的 `detect_pct_of_search_*` 欄位承載。兩者不可混用,
   也不可相加。
3. §3.1 描述 on-access 時,可加一句限定:兩種時機(載入期全掃 / access 時重算)在「查詢期
   記憶體不可變」下逐位元等價,而本文的 fault model 正是不可變假設不成立的場景——這是選擇
   on-access 的理由,不只是效率考量。
4. §3.3 引用 G 時必須保留 round-2 的限定語:「構造性」而非「偵測準確率」。

## Status of row 1 — what is and is not verified

Implemented and compiled; **the acceptance run has not been executed.** `--lazy-gate` asserts
L1 (load and lazy return identical top-k ids), L2 (count relation), and produces L3 (overhead).
It requires an **AVX512BW** host: RaBitQ's search path aborts at `utils/space.hpp:937` without
it, and the machine this was written on (i9-13900KF) has no AVX512 at all. Until that run is
green on the real workstation, row 1 is "implemented", not "measured", and §3 should not yet
quote an overhead number. Procedure: `artifacts/phase3/EXPB_RUNBOOK.md` §5.
