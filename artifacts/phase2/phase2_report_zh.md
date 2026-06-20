# Phase 2 報告：量化是否引入了 fp32 沒有的「災難性單點結構」？

> 本報告分析 `artifacts/phase2/` 在伺服器（workstation）上跑完的全量結果，回答 Phase 2 的主軸問題：
> **位元翻轉下，量化向量索引是否多出了一個全精度索引所沒有的「災難性單點崩潰結構」？若有，是哪一種量化、落在哪個位元組區域、能否便宜地偵測？**
> 資料來源：`vuln_map_pq.csv`（Tier 1）、`graph_edges_characterization.csv`（Tier 2）、`rollup/*.csv`（Tier 3）、`detection/*`（Tier 4）、`burst/*_burst.csv`（burst 子研究）。
> 本報告僅分析既有數據，**未修改任何實驗資料或程式碼**。設計依據為 `prompt.md`，執行方式見 `README.md`。

---

## 1. 摘要（TL;DR）

1. **「災難性單點結構」是 SQ8 獨有的，不是所有量化都有。** 只有 **scalar quantization 的 `sq_scale`**（1 KB 全域反量化參數）構成單一位元就能抹除整個 recall 的結構。fp32 索引沒有它；**PQ 也沒有**——這是本階段最反直覺的發現。

2. **PQ 的碼與 codebook 都不會災難性崩潰。** Tier 1 實測：`pq_codes`（離散索引位元）100% benign，與 SQ8 codes 一樣對單位元免疫；`pq_codebook` 雖在 exponent 位元有可量測的傷害（mean ΔR@10 ≈ 5e-4），但 **0% catastrophic**。PQ codebook 是**分散式**的——128 KB、32768 個子中心，沒有任一位元是全域命脈。**「量化更脆弱」對 PQ 不成立。**

3. **圖邊翻轉只會 crash 或無害，從不「沉默地」傷 recall。** Tier 2：HNSW 與 HNSW_SQ8 的 `graph_edges` 翻轉約 **71% 為 crash（可偵測）、29% silent-benign、0% silent-harmful**。這直接證實 `../first-try/` 的「圖損壞只影響速度、不影響 recall」，並說明為何主曲線必須把 graph_edges 排除——它是吸收翻轉的護盾，不是 recall 命脈。

4. **崩潰機率曲線（主結果）：fp32 ≈ 0、PQ ≈ 0、只有 SQ8 > 0。** Tier 3 Curve B 在 per-bit 率 `r` 上：FLAT / IVF_FLAT / HNSW / IVF_PQ 的 `cat_bits = 0 → P_collapse ≈ 0`；只有 **IVF_SQ8（cat_bits=2983）與 HNSW_SQ8（cat_bits=1406）** 有非零崩潰風險。「量化引入 vs fp32 沒有」這個主軸，精準地縮小成「**SQ8 引入了 sq_scale**」。

5. **range-check 護欄能便宜地擋掉一半到三分之二的災難。** Tier 4：值域檢查對 catastrophic 翻轉涵蓋 **53.5%**（HNSW_SQ8 sq_scale 達 63.8%），開銷僅一次查詢的 **0.01%–0.08%**，誤報率 4.9%。便宜但不完整——殘存在合法值域內的災難仍需校驗碼/冗餘。

6. **burst 暴露度與「結構是否分散」成反比。** worst-case 對齊注入下：**HNSW_SQ8 只要 64-bit burst 打進 sq_scale 就 100% 全崩**（ΔR=0.951）；**PQ codebook 即使被 262144-bit（整段）burst 打中也從不崩潰**（ΔR 僅 0.154，仍 > 50% 保留）；fp32 centroid 要 65536-bit 級 burst 才有輕微退化。集中的全域參數 = 脆弱；分散的結構 = 韌性。

**對核心問題的回答**：「量化是否更脆弱」**不能一概而論，取決於量化型態**。**SQ8 確實引入了 fp32 沒有的災難性單點結構（`sq_scale`）**；但 **PQ 沒有**——它的 codebook 分散到崩潰風險趨近於零。脆弱性的真正來源不是「量化」這個動作，而是「**是否存在被全體向量共用的微小全域參數**」。

---

## 2. 實驗設定與本階段的四條探針

延用 Phase 0/1 凍結的設計矩陣（IVF `nlist`=1024、圖 `M`=32 / `efC`=200、`k`=10、tolerant ε=0.02、seed=1234），各索引乾淨操作點同 Phase 1（`baseline.csv`）。Phase 2 加四條獨立探針：

| Tier | 腳本 | 問題 | 注入/分析對象 | 隔離 |
|---|---|---|---|---|
| 1 | `phase2_pq_sensitivity.py` | PQ 碼是否打破碼免疫？codebook 會災難嗎？ | IVF_PQ_M8 / M16 的 `pq_codes` / `pq_codebook` / `centroid` | in-process（乾淨反序列化） |
| 2 | `phase2_graph_sensitivity.py` | 圖邊翻轉 = crash / 無害 / 沉默有害？ | HNSW / HNSW_SQ8 的 `graph_edges` | **subprocess**（防 C++ segfault 殺掃描） |
| 3 | `phase2_rollup.py` | 合成兩條 faults/MB 曲線 | 純分析既有地圖，**不翻 bit** | — |
| 4 | `phase2_detection.py` | range-check 能擋多少災難？ | 重放 Phase 1 + Tier 1 受損實例 | — |
| burst | `phase2_burst.py` | 空間相關錯誤（壞 DIMM 區塊）暴露度？ | IVF_FLAT / IVF_PQ_M8 / HNSW_SQ8，worst-case 對齊臨界結構 | **subprocess** |

**基線模式（`baseline_mode`）**：對齊 0.95 的索引用**絕對** ΔR@10 > 0.01 定義 catastrophic（`aligned`）；PQ 因天花板達不到 0.95，用**相對**「保留率 < 50% of own clean」（`own`，M8 clean=0.379、M16=0.563）。兩種計數每列都存。

> **資料新鮮度**：Tier 3/4 依賴在新區域幾何（`codes_meta` 切分、含 IVF_SQ8 的 sq_scale）上重跑的 Phase 1 地圖。本批結果中 `curveB_collapse_prob.csv` 已含 IVF_SQ8 的 `cat_bits=2983`、detection 已含 IVF_SQ8 sq_scale catastrophic——代表**前置的 Phase 1 全量重跑已在伺服器完成**，Tier 3/4 建立在新幾何上，可信。

---

## 3. Tier 1 — PQ 單位元敏感度：碼免疫沒被打破，codebook 不會災難

`vuln_map_pq.csv`，IVF_PQ_M8 / M16 各三區域。關鍵列（相對自身 clean）：

| 索引 | 區域 | bit 位置 | n | mean ΔR@10 | max ΔR@10 | benign | **catastrophic(rel)** | silent-wrong |
|---|---|---|---:|---:|---:|---:|---:|---:|
| IVF_PQ_M8 | pq_codes | code_b0…b7 | 各125 | ≤4.8e-7 | ≤2.0e-5 | **100%** | **0%** | ≤6 |
| IVF_PQ_M16 | pq_codes | code_b0…b7 | 各125 | ≤5.6e-7 | ≤2.0e-5 | **100%** | **0%** | ≤7 |
| IVF_PQ_M8 | **pq_codebook** | **exponent** | 1000 | **3.45e-4** | 5.3e-3 | 47.3% | **0%** | 707 |
| IVF_PQ_M16 | **pq_codebook** | **exponent** | 1000 | **5.50e-4** | 4.6e-3 | 43.6% | **0%** | 745 |
| IVF_PQ_M8 | pq_codebook | sign | 1000 | 4.0e-5 | 1.1e-3 | 86.1% | 0% | 511 |
| IVF_PQ_M16 | pq_codebook | sign | 1000 | 7.9e-5 | 1.2e-3 | 78.1% | 0% | 604 |
| IVF_PQ_M8/M16 | pq_codebook | mantissa-low | 1000 | ≈0 | 0 | 100% | 0% | 0 |
| IVF_PQ_M8 | centroid | exponent | 500 | 1.25e-4 | 1.0e-3 | 53.8% | 0% | 356 |
| IVF_PQ_M16 | centroid | exponent | 500 | 1.83e-4 | 1.6e-3 | 47.6% | 0% | 386 |

**三個結論**：

1. **碼免疫成立。** `pq_codes` 是離散 codebook 索引（翻轉=跳到另一個子中心），原本假設「離散跳躍可能比 fp32 更糟」——**實測證偽**：100% benign、max ΔR 僅 2e-5，與 SQ8 codes 同級。把 pq_codebook「從推測變實測」的任務完成，答案是穩健。

2. **`pq_codebook` 可量測但不災難。** exponent 位元是 codebook 最敏感處（mean ΔR ≈ 5e-4，近半數 silent-wrong），但 **0% catastrophic**——沒有任何單位元能讓 PQ recall 掉到自身 clean 的一半以下。原因是 codebook 分散：128 KB / 32768 個子中心，每個子中心只服務碼空間的一小格，動一個位元只擾動「用到那個子中心的那一格碼」的向量，不是全體。**這與 SQ8 的 sq_scale（單一全域 scale）形成根本對比。**

3. **位元位置律與 Phase 1 完全一致**：exponent > sign > mantissa-high ≫ mantissa-low(=0)，符合 IEEE-754 數值權重。**全部 0 crash、0 NaN/Inf**——延續 Phase 1，傷害一律是 silent-wrong。

---

## 4. Tier 2 — graph_edges：只會 crash 或無害，0% 沉默有害

`graph_edges_characterization.csv`，每索引抽 4000 條邊（int32 節點 id，分層於高/低位元組）：

| 索引 | bit 位置 | n | **%crash** | **%silent-benign** | **%silent-harmful** | mean ΔR@10 | max ΔR@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| HNSW | ALL | 4000 | **70.7%** | 29.3% | **0.0%** | -1.7e-8 | 1.0e-5 |
| HNSW | id_high | 2000 | **89.2%** | 10.9% | 0.0% | 0 | 0 |
| HNSW | id_low | 2000 | 52.3% | 47.7% | 0.0% | -2.1e-8 | 1.0e-5 |
| HNSW_SQ8 | ALL | 4000 | **71.2%** | 28.8% | **0.0%** | 0 | 0 |
| HNSW_SQ8 | id_high | 2000 | **88.9%** | 11.1% | 0.0% | 0 | 0 |
| HNSW_SQ8 | id_low | 2000 | 53.6% | 46.5% | 0.0% | 0 | 0 |

**結論**：

- **沒有「沉默有害」這個分類。** 圖邊翻轉的結局只有兩種：**越界 id → C++ segfault → crash（可偵測）**，或**合法但錯的鄰居 → recall 幾乎不動（silent-benign）**。HNSW 圖的高連通性讓單一條錯邊被其他路徑繞過——這正是 `../first-try/` 「圖損壞只影響速度、不影響 recall」的位元級對應。
- **高位元組（id_high）→ 89% crash**：翻高位元組把 id 推到節點數之外，立刻越界；**低位元組（id_low）→ ~53% crash、~47% 無害**：小幅改 id 多半仍落在合法範圍，換到另一個存在的鄰居。
- **HNSW 與 HNSW_SQ8 幾乎逐列相同**（71% vs 71%、89% vs 89%）——印證 graph_edges 是**受控變因**（兩者共用同一圖結構），不是編碼效應。這就是 Tier 3 必須把它從主曲線剔除的根據：其 ~2.08×10⁹ bit 的巨大足跡若留在主曲線，會用「可偵測的 crash」淹沒 sq_scale 的量化訊號，還讓 fp32 HNSW 看似會崩潰。
- subprocess 隔離有效：crash 被記成 `crash` 而非殺掉整個掃描（`n_crash` 高達 2829/2849 但掃描完整跑完 4000 條）。

---

## 5. Tier 3 — faults/MB rollup（主結果）：崩潰風險只有 SQ8 非零

純分析，不翻 bit。在 per-bit 錯誤率 `r ∈ {1e-9 … 1e-5}` 上合成兩條曲線。

### 5.1 Curve B（主曲線）— P(catastrophic collapse)，**排除 graph_edges**

`curveB_collapse_prob.csv`。`cat_bits = catastrophic_fraction × region_bits`（臨界結構的絕對致命位元數），`P_collapse = 1 − (1−r)^cat_bits`。

| 索引 | baseline | **cat_bits** | P_collapse @ r=1e-7 | @ r=1e-6 | @ r=1e-5 | 驅動結構 |
|---|---|---:|---:|---:|---:|---|
| FLAT | aligned | **0** | 0 | 0 | 0 | （無） |
| IVF_FLAT | aligned | **0** | 0 | 0 | 0 | （無） |
| HNSW | aligned | **0** | 0 | 0 | 0 | （無） |
| **IVF_PQ_M8** | own | **0** | 0 | 0 | 0 | （無，codebook 分散） |
| **IVF_PQ_M16** | own | **0** | 0 | 0 | 0 | （無，codebook 分散） |
| **HNSW_SQ8** | aligned | **1406** | 1.4e-4 | 1.4e-3 | 1.40e-2 | **sq_scale** |
| **IVF_SQ8** | aligned | **2983** | 3.0e-4 | 3.0e-3 | 2.94e-2 | **sq_scale** |

**這就是本研究主軸圖的資料**：在崩潰機率這條曲線上，**fp32 三索引與 PQ 兩索引全部壓在 0**，只有兩個 **SQ8** 索引抬頭。命題「量化引入了 fp32 沒有的災難性單點結構」**被精準地限縮為「SQ8 引入了 `sq_scale`」**——PQ 雖也是量化，卻因 codebook 分散而與 fp32 同列於零風險。

兩個 SQ8 之間，**IVF_SQ8 的崩潰風險約為 HNSW_SQ8 的 2 倍**（cat_bits 2983 vs 1406）。雖共用同一個 scalar quantizer，IVF_SQ8 的 sq_scale 致命比例更高（見 region_terms：cat_frac 0.364 vs 0.172）——值域分布不同（IVF_SQ8 sq_scale ∈ [-123,258] vs HNSW_SQ8 ∈ [0,197]）使更多位元落在「翻了就致命」的範圍。

### 5.2 Curve B（graph_edges 獨立曲線）— 仍然是 0

`curveB_graph_edges.csv`：HNSW 與 HNSW_SQ8 的 graph_edges `cat_bits = 0 → P_collapse = 0`（因 Tier 2 測得 **0% silent-harmful**，而 crash 可偵測、不計入 collapse）。**圖結構對崩潰曲線零貢獻**——剔除它是正確的，且剔除後不影響任何索引的崩潰結論。

### 5.3 Curve A（附錄）— 期望 ΔRecall，IVF_SQ8 領先（足跡濃縮效應）

`curveA_expected_dR.csv`，線性疊加 Σ E(翻轉數)·E(ΔR|單翻轉)：

| 索引 | E[ΔR@10] @ r=1e-6 | @ r=1e-5 | 主導區域 |
|---|---:|---:|---|
| **IVF_SQ8** | 2.13e-3 | **2.13e-2** | sq_scale + centroid |
| IVF_FLAT | 1.02e-3 | 1.02e-2 | centroid |
| HNSW_SQ8 | 6.29e-4 | 6.29e-3 | sq_scale |
| FLAT | 5.69e-4 | 5.69e-3 | vectors |
| IVF_PQ_M16 | 4.17e-4 | 4.17e-3 | pq_codebook + centroid |
| HNSW | 2.65e-4 | 2.65e-3 | vectors |
| IVF_PQ_M8 | 2.61e-4 | 2.61e-3 | pq_codebook |

**驗證了 Phase 1 的 A4 假說**：IVF_SQ8 與 HNSW_SQ8 用**同一個 scalar quantizer**，但 IVF_SQ8 的期望傷害是 HNSW_SQ8 的 **3.4 倍**。差異不來自量化器，來自**記憶體足跡 profile**——IVF_SQ8 小而密（recall-relevant 近 100%），HNSW_SQ8 有 65% 圖結構當護盾吸收翻轉。`region_terms.csv` 佐證：IVF_SQ8 sq_scale 的 `E_dR=0.160`、HNSW_SQ8 sq_scale `E_dR=0.059`，sq_scale 是兩者 Curve A 的單一主導項。

> **正規化說明**：x 軸是**絕對每位元率 r**，可由 DRAM field-study（如 Schroeder 的 FIT/Mbit）換算成每位元每小時機率。崩潰風險由臨界結構的**絕對 bit 數**驅動（cat_bits），不是每次翻轉的條件機率——這是為什麼 1 KB 的 sq_scale 雖小卻是唯一非零來源。Curve A 僅在 `region_bits·r ≪ 1` 線性近似成立，超出已 clamp 到物理範圍 [-1,1]。

---

## 6. Tier 4 — range-check 偵測護欄：便宜、半涵蓋、需搭配冗餘

`detection/`。從**乾淨**索引算各 metadata 區域的訓練值域（`expected_ranges.json`），對受損 metadata 檢查是否越界（+ NaN/Inf 便宜附加）。重放 Phase 1 + Tier 1 的受損實例。

**整體涵蓋（`detection_summary.json`）**：

| 嚴重度 | n | 偵測到 | 涵蓋率 |
|---|---:|---:|---:|
| **catastrophic** | 4389 | 2348 | **53.5%** |
| moderate | 4516 | 2257 | 50.0% |
| benign（誤報） | 23479 | 1161 | **4.9% FP** |
| clean（誤報） | 5 | 0 | 0% |

**分區涵蓋（`coverage.csv`）**：

| 索引 | 區域 | catastrophic 涵蓋 | moderate 涵蓋 | benign 誤報 |
|---|---|---:|---:|---:|
| **HNSW_SQ8** | sq_scale | **63.8%**（897/1406） | 3.0% | 1.1% |
| **IVF_SQ8** | sq_scale | **48.6%**（1451/2983） | 1.4% | 0.4% |
| IVF_FLAT | centroid | — | 98.2% | 3.4% |
| IVF_SQ8 | centroid | — | 77.0% | 2.5% |
| IVF_PQ_M8 | centroid | — | 94.0% | 29.3% |
| IVF_PQ_M16 | centroid | — | 94.9% | 27.9% |
| IVF_PQ_M8 | pq_codebook | — | 75.4% | 0.5% |
| IVF_PQ_M16 | pq_codebook | — | 66.4% | 0.1% |

**開銷**：guard 掃 metadata 的時間是一次查詢的 **0.01%–0.08%**（如 HNSW_SQ8：7.5µs vs 11.9ms = 0.06%）——幾乎免費。

**結論**：

- **range-check 是高 CP 值的第一道防線**：免費、低誤報（4.9%），就能擋掉 **53.5% 的災難性翻轉**（HNSW_SQ8 sq_scale 達 63.8%）。值得標配。
- **但它擋不滿。** 沒被擋下的 catastrophic 翻轉，是那些把值改成**仍落在合法 [min,max] 值域內、卻已足以毀掉 recall** 的翻轉（典型如 exponent 在量級內的位移、某些 sign 翻轉）。值域檢查只攔「越界」，攔不了「界內的錯」。
- **延續 Phase 1 的鐵律**：傷害全是 silent-finite-wrong，被動偵測有上限。要逼近 100% 涵蓋，sq_scale 這種 1 KB 級全域參數必須上**校驗碼 / 三重冗餘**（成本可忽略，見第 8 節）。range-check 是便宜的補強，不是替代。
- moderate 的 centroid 涵蓋反而高（77–98%）——centroid 致命幅度溫和、越界比例高，較易被值域抓到；PQ centroid 誤報率偏高（~28%），因 PQ 的粗量化 centroid 值域較緊。

---

## 7. Burst 子研究：暴露度與「結構是否分散」成反比

`burst/*_burst.csv`，`--aligned-worstcase`（每次 burst 起點對齊該索引最小的臨界結構），每個 burst 長度 B 跑 50 trials，subprocess 隔離。

| B (bits) | **HNSW_SQ8 / sq_scale** | **IVF_PQ_M8 / pq_codebook** | **IVF_FLAT / centroid** |
|---:|---|---|---|
| 1 | ΔR=0, p_collapse=0 | ΔR=0, p=0 | ΔR=0, p=0 |
| 8 | ΔR=0, p=0 | ΔR=0, p=0 | ΔR=0, p=0 |
| **64** | **ΔR=0.951, p=1.0** 💥 | ΔR=4e-4, p=0 | ΔR≈0, p=0 |
| 512 | ΔR=0.951, p=1.0 | ΔR=6e-4, p=0 | ΔR=5e-5, p=0 |
| 1024 | ΔR=0.951, p=1.0 | ΔR=1.2e-3, p=0 | ΔR=9e-5, p=0 |
| 8192 | ΔR=0.951, p=1.0 | ΔR=6.3e-3, p=0 | ΔR=1.6e-3, p=0 |
| 65536 | **50 crash**（溢出至圖/header） | ΔR=4.2e-2, p=0 | ΔR=0.019, **p=1.0** |
| 262144 | 50 crash | **ΔR=0.154, p=0** | ΔR=0.057, p=1.0 |

**結論（暴露度的物理直覺）**：

- **HNSW_SQ8 sq_scale —— 一觸即潰。** 只要 **64-bit（8 bytes）burst** 落在 1 KB 的 sq_scale，**每一次 trial 都全崩**（ΔR=0.951）。B≥65536 時 burst 溢出 sq_scale 邊界打進圖/header → 50/50 crash。這是「微小全域參數」的最壞情境：極小的空間相關錯誤就足以摧毀整個索引。
- **PQ codebook —— 幾乎打不倒。** 即使 **262144-bit（32 KB，佔整段 128 KB codebook 的 1/4）** worst-case burst，relative collapse 仍是 **0**，ΔR 只到 0.154（相對 own clean 0.379 仍保留 > 50%）。分散式結構把集中的物理錯誤稀釋掉了。**0 crash**——codebook 是純資料區，不含會 segfault 的指標。
- **fp32 centroid —— 要很大的 burst 才有輕微效果。** 512 KB 的 centroid 要到 65536-bit 級 burst 才 p_collapse=1.0，但 ΔR 也僅 0.019–0.057（溫和）；單一 centroid 只影響它那條 inverted list。**0 crash**。

**主軸印證**：burst 暴露度 ∝ 1/（臨界結構的分散程度）。sq_scale（1 KB、全域共用）≫ centroid（512 KB、分片）≫ pq_codebook（128 KB、分散到崩潰免疫）。**「量化把足跡縮小」本身不危險；危險的是 SQ8 把整個解碼壓在一個 1 KB 的全域 scale 上。**

---

## 8. 綜合結論與對 Phase 3 的建議

### 8.1 對主問題的最終回答

> **量化是否讓向量索引在位元翻轉下更脆弱？——取決於量化型態，不是「量化」與否。**

| 索引族 | 災難性單點結構 | 崩潰曲線 (Curve B) | burst 韌性 | 判定 |
|---|---|---|---|---|
| fp32（FLAT/IVF_FLAT/HNSW） | 無 | ≈ 0 | 高 | 基準 |
| **PQ**（IVF_PQ_M8/M16） | **無**（codebook 分散） | **≈ 0** | **高**（打不倒） | **與 fp32 同級，未更脆弱** |
| **SQ8**（IVF_SQ8/HNSW_SQ8） | **有：`sq_scale`** | **> 0**（唯一非零） | **極低**（64-bit 即崩） | **唯一被量化引入的災難結構** |

脆弱性的根因不是「量化」這個動作，而是「**是否存在一個被全體向量共用、卻只佔 KB 級記憶體的全域反量化參數**」。SQ8 有（`sq_scale`），PQ 沒有（codebook 分散到 32768 個局部子中心）。這修正了「量化 = 更脆弱」的籠統直覺。

### 8.2 防護建議（按 ROI 排序）

1. **對 SQ8 `sq_scale` 上校驗碼 / 三重冗餘** —— 最高 ROI。1 KB 級、成本可忽略，卻封住整個研究唯一的災難性單點結構（單位元 max ΔR 0.951、64-bit burst 必崩）。
2. **range-check 護欄標配** —— 免費（< 一次查詢的 0.1%）、低誤報（4.9%），擋掉 53.5% 災難。作為 #1 之外的便宜第二層，攔截越界型損毀。
3. **PQ / fp32 不需單點硬化** —— Curve B ≈ 0、burst 打不倒，把保護預算省下來。
4. **graph_edges 靠既有 crash 偵測即可** —— 0% silent-harmful，越界即 segfault；不需額外 recall 護欄（呼應 `../first-try/` 與 Q3 冗餘比較）。

### 8.3 Phase 3 銜接

- **HNSW_PQ**（本階段 PQ 只做 IVF 族，HNSW_PQ 留 Phase 3）：驗證「PQ 無災難結構」在圖索引上是否仍成立。
- **GIST1M / 文字嵌入**（尚未下載）：跨資料集泛化 sq_scale 災難與 PQ 韌性。
- **Q3 冗餘比較**：把 #1（sq_scale 冗餘）的成本/效益接回 `../first-try/` 的圖冗餘討論，形成「哪種結構值得冗餘」的統一結論。

---

## 附錄：資料來源

| 檔案 | 內容 | 對應章節 |
|---|---|---|
| `vuln_map_pq.csv` / `.json` | Tier 1：PQ 單位元敏感度（index×region×bit_tag 聚合） | §3 |
| `graph_edges_characterization.csv` / `.json` | Tier 2：圖邊翻轉三分類（crash/benign/harmful） | §4 |
| `rollup/curveB_collapse_prob.csv` | Tier 3 主曲線：P(collapse)(r)，排除 graph_edges | §5.1 |
| `rollup/curveB_graph_edges.csv` | Tier 3：graph_edges 獨立崩潰曲線（=0） | §5.2 |
| `rollup/curveA_expected_dR.csv` / `curveA_graph_edges.csv` | Tier 3 附錄：期望 ΔRecall(r) | §5.3 |
| `rollup/region_terms.csv` | 每 (index×region) 的 E_dR / cat_frac / series 歸因 | §5 |
| `detection/coverage.csv` / `detection_summary.json` / `expected_ranges.json` | Tier 4：range-check 涵蓋率/誤報/開銷/訓練值域 | §6 |
| `burst/{IVF_FLAT,IVF_PQ_M8,HNSW_SQ8}_burst.csv` | burst 子研究：worst-case 對齊掃 B | §7 |

*本報告僅分析 `artifacts/phase2/` 既有全量數據，未修改任何實驗資料或程式碼。生成腳本見 `artifacts/phase2/README.md`。基線與 Phase 1 地圖見 `../baseline.csv`、`../phase1/phase1_report_zh.md`。*
