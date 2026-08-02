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

## §3.4 / §4 — cost accounting

| # | Claim | Implementation | Level |
|---|---|---|---|
| 14 | **Tiered cost: 132 B catastrophe protection + 0.03–5.7% detection layer** (§3.4, abstract) | `phase3_f_synth.py` → `frontier.csv` `cliff_protection_bytes` / `slope_manifest_bytes` / `protection_pct_of_index` | **measured** for the *file* sizes (132 B and 16 000 064 B = 5.7027% of the 280 573 456 B index). **But the 5.7% upper bound is a serialization number, not a residency number** — see "Manifest file format ≠ resident cost" below. The honest resident range is **0.03–1.4%**. |

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

### One nuance the measurement does NOT resolve

The mechanism argument above ("EB's bytes are read anyway") predicts EB should pay a lower cost
*per byte*, not merely fewer bytes. Converting the headline deltas to per-byte rates:

| | ns per CRC'd byte |
|---|---|
| load-time scan (sequential, whole 96 MB) | 1.267–1.269 (two independent runs) |
| `fallback_eb` (on-access) | 1.511 |
| `drop` (on-access) | 1.651 |

EB is only **8.5%** cheaper per byte than `drop`, and the other cross-check disagrees on the sign
(`timed_ns_per_check`: EB 193.96 vs drop 186.95). **Two methods that disagree on the sign of an
8.5% effect have not measured it.** So: the count ratio (26.8×) is resolved and load-bearing; the
cache-locality saving that "搭便車" names is *not*. The likely reason is that at 96 B a
byte-at-a-time software CRC is compute-bound, not bandwidth-bound, so whether the bytes were
already resident barely matters.

Wording consequence: say EB is cheap **because it checks 26.8× fewer times**, not because its
bytes are already in cache. The latter is mechanically true and unmeasured.

## Manifest file format ≠ resident cost (the 5.7% is not what the detector holds)

The manifest entry is 16 B — `{u64 abs_file_offset, u32 len, u32 crc32}` (`crc_manifest.py`
header spec) — which is where 16 MB / **5.7027%** comes from. But `off` and `len` are re-derivable
from the geometry, and the implementation does exactly that: it re-derives them, uses them only to
*verify* the manifest against the index, and then throws them away.

```cpp
std::vector<Entry> entries(n_entries);   // hnsw.hpp:293 — a LOCAL; freed when the function returns
...
    fi_crc_expect_[i] = entries[i].crc;  // hnsw.hpp:323 — only the 4-byte CRC survives
```

Resident structures after `load_crc_manifest()` returns:

| structure | bytes/element | total | % of index | load | lazy |
|---|---|---|---|---|---|
| `ex_corrupt_` — eager verdict bitmap (`hnsw.hpp:69`) | 1 | 1 MB | 0.356% | ✅ | allocated but unused |
| `fi_crc_expect_` — expected CRCs (`:138`) | 4 | 4 MB | **1.426%** | — | ✅ |
| `fi_crc_seen_fail_` — distinct-count stats only (`:146`) | 1 | 1 MB | 0.356% | — | ✅ (droppable in production) |

So **on-access detection's true resident cost is 4 B/element = 1.43% of the index**, which is
exactly the "CRC32/元素為 1.4%" rung the paper's own granularity ladder already names. The 16 B
figure is the on-disk serialization, carrying redundant fields whose only job is to make a
mispaired manifest fail loudly rather than silently.

**Wording consequence:** the abstract's 「偵測層開銷依 CRC 粒度為 0.03–5.7%」 quotes a file-format
number as its upper bound. The resident range for the same granularity ladder is **0.03–1.4%**.
This correction is in the paper's favour and needs no implementation change — but do not quietly
swap the number without saying which quantity is being reported, since 5.7% is the right answer if
what you mean is "bytes that must be persisted alongside the index".

## Detection cost relative to NO detection

`headline_pct_of_search` in `expb_lazy_gates.json` is measured against the **load** arm — i.e.
detection is on in both arms and only the CRC's timing moves. That is the right denominator for
"what does moving the CRC to the query path cost", but it is *not* "what does detection cost".
The second question needs `recovery=none`, which `--lazy-gate` never runs.

Measured directly on the clean index (all five configs produce byte-identical dumped ids, so the
search work is identical and every delta is pure detection cost):

| config | µs/query | vs `none` | CRC checks | CRC bytes | load-time scan |
|---|---|---|---|---|---|
| `none` | 1 433.98 | — | 0 | 0 | 0 |
| `fallback_eb` + load | 1 424.83 | −0.64% (noise floor) | 0 | 0 | 118.6 ms |
| `drop` + load | 1 466.92 | +2.30% | 0 | 0 | 118.7 ms |
| `fallback_eb` + lazy | 1 545.60 | **+7.78%** | 7 549 690 | 724.8 MB | 0 |
| `drop` + lazy | 4 438.68 | **+209.53%** | 204 517 241 | 19.63 GB | 0 |

Three readings:

1. **Eager detection is genuinely near-free on the query path** — EB below the noise floor, drop
   +2.30%. That +2.30% has a clean account: 20 634 `ex_corrupt_[id]` bitmap lookups per query,
   33 µs / 20 634 = **~1.6 ns per lookup**, the cost of one cached byte load. Its real price is the
   one-off 118.7 ms scan — and the staleness that motivated this whole branch.
2. **The `none` and `load` denominators give nearly the same answer** (drop: +209.53% vs +224.24%),
   because `load` is itself only +2.30% over `none`. The remaining gap is that the headline run
   used the *corrupted* index, where drop actually drops nodes (206.3 M vs 204.5 M consults).
3. Per neighbour: search is 1 433.98 µs / 20 451.7 consults = **70.1 ns per neighbour evaluation**;
   drop's on-access CRC adds **146.9 ns**. The CRC is **2.10× the entire per-neighbour search cost.**

> **Provenance caveat.** Unlike every other number in this document, this table was produced by
> invoking `exp_dumpids` directly, not by a driver, so it has no program-stamped provenance and no
> artifact under `artifacts/phase3/`. Reproduce with, on `_expb_clean.index` + `expb_clean_ex_code.crcmf`:
> `exp_dumpids <index> <query> <gt> l2 2000 <out> 10 --recovery {none|drop|fallback_eb} [--crc-manifest <mf>] [--crc-mode lazy] --stats-json <json>`.
> **Promote it to a driver + test before §3 or §4 cites it.**

## Optimization headroom (why the drop number is not a law of nature)

`fi_crc32` (`hnsw.hpp:208`) is a **byte-at-a-time table-driven** CRC-32/ISO-HDLC — one table lookup
and XOR per byte, with a loop-carried dependency on `crc`. Measured 1.53 ns/B ≈ 8.7 cycles/byte,
which is what that formulation costs. It is the slowest common implementation, so the 224%/209%
figures characterise *this* kernel, not on-access detection in general.

| option | polynomial-compatible? | manifest change | extra memory | 96 B compute (projected) |
|---|---|---|---|---|
| PCLMULQDQ folding | ✅ same 0xEDB88320 | **none** — stays bit-identical to `zlib.crc32` | none (no table) | ~15–25 ns |
| SSE4.2 `_mm_crc32_u64` | ❌ CRC32**C** (0x82F63B78) | new `algo` tag + regenerate | none (no table) | ~6–10 ns |
| slicing-by-8 | ✅ same | none | +7 KB constant table | ~17 ns |

The manifest header already reserves `u32 algo` at offset 12 (`ALGO_CRC32 = 1`), so a second
algorithm was anticipated by the format. PCLMULQDQ is nonetheless the more attractive route: it
keeps byte-identical compatibility with every frozen manifest and record, so nothing needs
re-running.

**But the ceiling is memory, not CRC.** Once the kernel is fast, what remains for `drop` is the
1.98 MB/query of ex data it pulls in that the 1-bit traversal would never have touched.

### ✅ Now measured — `rabitq_instrumentation/qp_crc32.hpp`, see `docs/crc_kernel.md`

Both projections above were tested by implementing PCLMULQDQ folding (and slicing-by-8 as a
control) and re-measuring. Median of 3 interleaved rounds, clean index, ef=2000, baseline
`--recovery none`:

| policy | table (was) | slice8 | **clmul** | projected | held? |
|---|---|---|---|---|---|
| `fallback_eb` | +8.51% | +3.55% | **+1.6–2.5%** | ~1% | close, slightly optimistic |
| `drop` | +209.83% | +76.92% | **+49.92%** | 40–70% | **yes, mid-range** |

Load-time scan (sequential CRC over the 96 MB ex region): 112.9 ms → **9.8 ms**, a **11.5×**
kernel speedup (10.8 GB/s on the real index). Every kernel is bit-identical to `zlib.crc32`, and
the acceptance gates re-run on `clmul` came back green with **recall identical to the last digit**
(EB 0.94329, drop 0.94050) and bit-identical ids.

**The EB claim is now unresolvable rather than small.** Five repeats put EB's on-access cost at
+1.62% with a within-arm spread 3.7× the delta. The right phrasing for §3 is "not resolvable
against run-to-run variation", not a precise small percentage. Note `--lazy-gate` runs each arm
once and reported **−1.73%** for EB on this build — that is noise, not a finding, and its EB
headline should now be read as "below resolution".

**`drop`'s remaining 50% is 75% memory, 25% compute.** From the acceptance run's own cross-check:
`analytic` (crc_bytes × the *sequential* scan rate) = 182 µs/query vs a measured headline of
736 µs/query. The 4× gap is the random-access penalty. ⚠️ **This also means the `analytic`
cross-check is only valid in the compute-bound regime** — it tracked headline to within 23% with
the table kernel and underestimates by 4× with `clmul`.

So hardware CRC **makes the piggyback claim true for EB** and **does not rescue `drop`** — now
with numbers behind both halves. The structural fix for `drop` is the consult site (option B),
not a faster kernel.

### Not a legitimate optimization

Memoizing verdicts across queries would eliminate nearly all of the cost — and would re-introduce
precisely the staleness this branch exists to remove, since the fault model has errors accumulating
during residency. `ex_corrupted_now()` says so at `hnsw.hpp:163`: *"Recomputed every call by
design."* Within a single query there is nothing to memoize either: `drop` consults only *unvisited*
neighbours, so each element is already checked at most once per query.

## Consequence: EB dominates `drop` on every measured axis

Worth stating plainly, because it is now a stronger claim than when only recall was on the table:

| axis | `fallback_eb` | `drop` | source |
|---|---|---|---|
| recall, severe damage (384 flips/elem) | **0.81425** | 0.79597 | `expb_gates.json` gate 4 |
| recall, light damage (4 flips/elem) | **0.81632** | 0.79739 | `expb_gates.json` gate 4 (informational) |
| tolerable corruption f\* @ R≥0.90 | **0.1006** | 0.0926 | `frontier.csv` |
| tolerable corruption f\* @ R≥0.60 | **0.4650** | 0.3995 | `frontier.csv` |
| on-access detection cost | **+7.78%** | +209.53% | table above |

Recall and f\* separated the two policies only modestly (`interval_ratio` 1.06–1.23). A **27×**
cost axis does not. §3 can reasonably present EB as the recommended policy and `drop` as the
ablation, rather than as two co-equal options.

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
5. **記憶體那一根軸的上界要改**:5.7% 是 manifest 的**磁碟格式**(16 B/元素,含可推導的
   off/len);偵測器**常駐**的只有 4 B/元素 = **1.43%**,即階梯上原本就寫著的「1.4%」那一階。
   若講的是「必須與索引一起持久化的位元組」,5.7% 正確;若講的是「偵測層佔用的記憶體」,
   應為 **0.03–1.4%**。換數字時務必說明換的是哪一個量。見「Manifest file format ≠ resident cost」。
6. **成本數字要標明 CRC 核心**,而且**現在必須引用 `clmul` 的數字,不是 table 的**:
   +7.78% / +209.53% 是逐位元組查表核心(1.53 ns/B)的成本,不是 on-access 偵測的固有成本。
   換成 PCLMULQDQ(同一個多項式、逐位元相同)後實測:
   **EB 已量不出來**(+1.62%,而 run 間變異是該差值的 3.7 倍 —— 措辭應為「無法從一般執行變異中
   分辨」,而非引用一個看似精確的小百分比);**drop 為 +49.92%**,其中約 75% 是記憶體流量、
   25% 才是計算。**若 §3 要主張偵測便宜,必須同時綁定 EB 政策與快速核心。**
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
