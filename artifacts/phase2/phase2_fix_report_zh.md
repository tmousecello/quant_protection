# Phase 2 修正報告：把「collapse（崩潰）」門檻重新定義成單一標準

> **狀態：已實作並驗證完成（2026-06-14）。** 本報告記錄一個會讓主圖**自相矛盾**的定義問題、修正方案，以及修正後的**最終數據**。
>
> **取代聲明**：本報告承載修正後的 collapse 數據（cat_bits、burst p、cat_frac 等）。基準報告
> [`phase2_report_zh.md`](phase2_report_zh.md) 維持**修正前的原始記錄**（已 commit 存檔，刻意不改），其
> collapse 相關數字（cat_bits 2983/1406、fp32 burst p=1.0）**以本報告為準、已被取代**。實驗資料檔（
> `rollup/curveB_collapse_prob.csv`、`burst/*_burst.csv`、`vuln_map*.csv` 等）已重算，與**本報告**一致。

---

## 1. 問題：同一個詞、兩個門檻

修正前，「collapse / catastrophic」被兩套標準同時使用：

| 使用處 | 實際門檻 | 語意 |
|---|---|---|
| `phase2_burst.is_collapse`（aligned 分支）、Curve B 的 `cat_frac` | **ΔR@10 > 0.01** | 其實是「**有害（harmful）**」低標 |
| PQ（`baseline_mode="own"`） | **保留率 < 50% of own clean** | 這才是「**崩潰（collapse）**」 |

把「有害」當成「崩潰」，在 **fp32** 身上炸出一個明顯破綻。

### 1.1 破綻（具體數字）

- **burst 表**（`burst/IVF_FLAT_burst.csv`）：fp32 的 `IVF_FLAT` 在 B=65536 / 262144 顯示 **`p_collapse = 1.0`**——但實際傷害只有 **ΔR@10 = 0.019 / 0.057**。它被標成「崩潰」純粹因為 `0.019 > 0.01`。
- **Curve B 主圖**（`rollup/curveB_collapse_prob.csv`）：同一個 `IVF_FLAT` 的 **`P_collapse = 0.0`**，所有 `r` 皆然。

於是讀者會同時看到「**fp32 永不崩潰**」（主圖）與「**fp32 burst → 100% 崩潰**」（burst 表）——同一個索引、兩個相反結論。修正前的 `phase2_report_zh.md`（約第 89 行）是用一段文字「解釋掉」這個矛盾，而非真正消除。

### 1.2 根因

`is_collapse` 的 aligned 分支用 `ΔR > buckets.CATASTROPHIC_ABS (0.01)`，PQ 用保留率 50%。**兩個門檻、一個名字**。唯一的修正：**定義單一 collapse 門檻，全索引族一致套用。**

---

## 2. 設計決策（已與使用者確認）

| # | 決策 | 內容 |
|---|---|---|
| 1 | **定義** | 統一採「**保留率 < 50% of own clean recall@10**」，全索引族一致（`config.COLLAPSE_RETENTION_FRAC = 0.5`，即 PQ 原本就用的值）。 |
| 2 | **crash 歸類** | collapse 只算「**沉默崩潰（silent）**」；可偵測的 crash / nan-inf **另計**，不算 collapse（與 Curve B 既有作法、以及本專案「沉默 vs 可偵測」的主軸一致）。 |
| 3 | **傳播範圍** | **全面一致**：Curve B 的 SQ8 `cat_frac` 也用新規則重算，不只修 fp32 的 burst。 |

兩層詞彙（修正後固定）：**harmful** = ΔR@10 > 0.01（`buckets.HARMFUL_ABS`，給單位元敏感度 / Curve A / detection）；**collapse** = 保留率<50%（僅沉默，給主圖 Curve B + burst）。判定統一走 `qp.metrics.is_silent_collapse`。

---

## 3. 讓「全面一致」變便宜的關鍵推論

一個保留率崩潰的翻轉必然 ΔR > ~0.475 > 0.01，所以：

> **collapse ⊆ harmful（ΔR>0.01）**

因此**凡是現行 `pct_catastrophic`(harmful) 已是 0 的區域，新規則下 `pct_collapse` 必為 0**（邏輯蘊含，不需重測）。修正前的 `region_terms.csv` 中，`cat_frac` 非零的區域**只有** `IVF_SQ8 / sq_scale`（當時 harmful-based cat_frac = 0.364）與 `HNSW_SQ8 / sq_scale`（0.172）；其餘（vectors、codes、centroid、pq_codebook、graph_edges）全為 0。

**結論：整個 Curve B 重算縮小成「只重掃 `sq_scale`」**——8192 bit × 2 個 SQ8 索引，可完整列舉、反序列化乾淨、in-process 快速。**完全不必重跑** Phase 1 / Tier 1 / Tier 2 的大規模注入；全部在本機完成（`artifacts/indexes/`、`sift/`、`gt/`、`baseline.json` 皆在）。

---

## 4. 已完成的修改

| 檔案 | 修改 |
|---|---|
| `qp/config.py` | 新增 `COLLAPSE_RETENTION_FRAC = 0.5`（`PHASE2_PQ_RETENTION_FRAC` 保留為別名）。 |
| `qp/buckets.py` | 保留 `CATASTROPHIC_ABS=0.01`，新增 `HARMFUL_ABS` 別名與兩層詞彙 docstring。 |
| `qp/metrics.py` | 新增唯一真理來源 `is_silent_collapse(faulted10, clean10, frac)`（crash/None → False）。 |
| `phase2_burst.py` | `is_collapse` 改用 `metrics.is_silent_collapse`、移除 baseline_mode 分支；欄位 `p_collapse`→`p_silent_collapse`；crash 另計。已重跑 `--aligned-worstcase --trials 50`。 |
| `phase1_sensitivity.py`、`phase2_pq_sensitivity.py` | aggregator 新增原生 `pct_collapse` 欄（未來全量重跑自動帶此欄，不再需要本次的補掃）。 |
| `phase2_recompute_collapse.py`（新增） | 列舉重掃 `sq_scale`（兩個 SQ8、8192 bit），寫入 `pct_collapse` 至 `vuln_map*.{csv,json}` 與 `rollup/sq_scale_collapse.csv`，並落地 per-flip raw jsonl。 |
| `phase2_rollup.py` | `cat_frac` 改由 `pct_collapse` 取（缺欄回退 `pct_catastrophic`）。已重跑。 |
| `phase2_detection.py` | `severity` 標籤 `catastrophic`→`harmful`，新增平行的 `collapse` 涵蓋指標；既有 `detection/{summary.json,coverage.csv}` 已對應改標籤。 |
| `README.md`、`prompt.md` | 更新 collapse 詞彙與 B6 決策註記。 |

> `phase2_report_zh.md` **未修改**（保留為原始記錄）。

---

## 5. 修正後的數據

### 5.1 Curve B（主曲線）— 重算前 → 重算後

`collapse_fraction`（= `pct_collapse`，保留率規則）比舊的 harmful(>0.01) 嚴格，故 cat_bits 變小、但 **fp32 與 PQ 仍為 0、SQ8 仍 > 0**——主結論不變。

| 索引 | cat_bits（舊·harmful） | **cat_bits（新·collapse）** | P_collapse @1e-7 | @1e-6 | @1e-5 |
|---|---:|---:|---:|---:|---:|
| FLAT / IVF_FLAT / HNSW | 0 | **0** | 0 | 0 | 0 |
| IVF_PQ_M8 / M16 | 0 | **0** | 0 | 0 | 0 |
| **HNSW_SQ8** | 1406 | **512** | 5.1e-5 | 5.1e-4 | 5.11e-3 |
| **IVF_SQ8** | 2983 | **1185** | 1.2e-4 | 1.2e-3 | 1.18e-2 |

`region_terms.csv` 的 sq_scale `cat_frac`：HNSW_SQ8 0.172 → **0.0625**；IVF_SQ8 0.364 → **0.145**。兩 SQ8 之間 IVF_SQ8 仍約為 HNSW_SQ8 的 **2.3 倍**（1185 vs 512）。

### 5.2 sq_scale 逐 tag：pct_catastrophic(harmful) → pct_collapse

重掃結果（`rollup/sq_scale_collapse.csv`）。可見「harmful 但未到 collapse」的位元被正確排除：HNSW_SQ8 的 sign、兩者的 mantissa-high 從非零 harmful 掉到 0% collapse。

| 索引 | tag | n | pct_catastrophic(舊) | **pct_collapse(新)** |
|---|---|---:|---:|---:|
| IVF_SQ8 | exponent | 2048 | 100% | **56.2%** |
| IVF_SQ8 | sign | 256 | 100% | **13.7%** |
| IVF_SQ8 | mantissa-high | 3840 | 17.7% | **0%** |
| IVF_SQ8 | mantissa-low | 2048 | 0% | 0% |
| HNSW_SQ8 | exponent | 2048 | 50% | **25.0%** |
| HNSW_SQ8 | sign | 256 | 50% | **0%** |
| HNSW_SQ8 | mantissa-high | 3840 | 6.6% | **0%** |
| HNSW_SQ8 | mantissa-low | 2048 | 0% | 0% |

### 5.3 Burst（`p` = p_silent_collapse；crash 另計）— 破綻消除

`burst/*_burst.csv`，`--aligned-worstcase --trials 50`。**關鍵變化：fp32 的 `IVF_FLAT` 由舊版 p=1.0（@B≥65536）變成「每個 B 都 0」**；HNSW_SQ8 大 B 的「50 crash」由舊版誤計成 collapse 改為**可偵測 crash、p=0**。

| B (bits) | HNSW_SQ8 / sq_scale | IVF_PQ_M8 / pq_codebook | IVF_FLAT / centroid |
|---:|---|---|---|
| 1 | ΔR=0, p=0 | ΔR=0, p=0 | ΔR=0, p=0 |
| 8 | ΔR=0, p=0 | ΔR=0, p=0 | ΔR=0, p=0 |
| **64** | **ΔR=0.951, p=1.0** 💥 | ΔR=2e-4, p=0 | ΔR≈0, p=0 |
| 512 | ΔR=0.951, p=1.0 | ΔR=5e-4, p=0 | ΔR=1e-4, p=0 |
| 1024 | ΔR=0.951, p=1.0 | ΔR=1.1e-3, p=0 | ΔR=1e-4, p=0 |
| 8192 | ΔR=0.951, p=1.0 | ΔR=5.7e-3, p=0 | ΔR=1.6e-3, p=0 |
| 65536 | 50 crash（可偵測）, **p=0** | ΔR=4.2e-2, p=0 | ΔR=0.019, **p=0**（舊=1.0） |
| 262144 | 50 crash, **p=0** | ΔR=0.158, p=0 | ΔR=0.057, **p=0**（舊=1.0） |

### 5.4 一致性檢查（本次目的）— PASS

| 索引族 | Curve B | Burst（max p_silent_collapse） | 一致？ |
|---|---|---|---|
| fp32（IVF_FLAT 代表） | 0 | 0（所有 B） | ✅ |
| PQ（IVF_PQ_M8 代表） | 0 | 0（所有 B） | ✅ |
| SQ8（HNSW_SQ8 / IVF_SQ8） | >0（cat_bits 512 / 1185） | 1.0（B=64–8192，沉默） | ✅ |

**沒有任何索引「一張圖崩、另一張圖不崩」。** fp32 與 PQ 在兩圖都 0；SQ8 在兩圖都呈崩潰。破綻消除。

### 5.5 detection（Tier 4）詞彙

`severity` 的 `catastrophic` 標籤改為 `harmful`（既有 53.5% 涵蓋率本就是 >0.01 的 harmful 涵蓋）。新增的 `collapse`（保留率<50%）子集涵蓋率需要重放原始 per-flip 記錄重跑——本機未保留 raw 記錄，故此欄待伺服器重跑補上；因 `collapse ⊆ harmful` 且崩潰多半把值推出合法值域，預期 ≥ 53.5%。

---

## 6. 驗證（已執行）

1. `qp.metrics.is_silent_collapse` 單元行為：`0.4/0.95→True`、`0.89/0.95→False`、`None→False`。✅
2. smoke：`phase2_burst.py --smoke`、`phase2_rollup.py --smoke`、`phase2_recompute_collapse.py --smoke`、`phase2_detection.py --smoke` 全通過。✅
3. 全量本機重跑：`phase2_recompute_collapse.py --queries 1000`（重掃 sq_scale）→ `phase2_rollup.py` → `phase2_burst.py --aligned-worstcase --trials 50`。✅
4. 9 個腳本 `py_compile` 全通過。✅
5. §5.4 一致性檢查腳本：**PASS**。✅

> **後續（非阻塞）**：若要讓 `phase2_report_zh.md` 主報告也呈現修正後數字，可依本報告 §5 的表格更新；目前刻意保留其為原始版本，以本修正報告為準。
