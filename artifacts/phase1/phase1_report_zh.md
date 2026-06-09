# Phase 1 報告：向量索引單位元翻轉敏感度地圖

> 本報告分析 `artifacts/phase1/` 既有實驗數據，回答本研究的核心問題：**當記憶體中的向量索引發生位元翻轉（DRAM soft error / Rowhammer / 儲存損毀）時，量化索引是否比全精度索引更容易掉 recall？**
> 資料來源：`vuln_map.csv`、`regions_aug/*.regions.json`、`../baseline.csv`、`../region_accounting.md`。本報告僅做分析，未改動任何資料或程式碼。

---

## 1. 摘要（TL;DR）

1. **量化的「碼 (codes)」本身對單位元翻轉幾乎免疫。** FLAT / HNSW 的 fp32 向量、HNSW_SQ8 的 SQ8 codes、IVF_FLAT 的 codes——所有「大區域」的 max ΔRecall@10 都 ≤ 2e-5，100% benign。**SQ8 量化碼並沒有比 fp32 向量更脆弱。**

2. **真正的脆弱點集中在「微小的共享元資料」。** SQ8 的 `sq_scale`（僅 1 KB）與 IVF 的 `centroid`（512 KB）是少數會造成大幅 recall 崩塌的區域。最嚴重者為 **HNSW_SQ8 `sq_scale` 的指數位元**：mean ΔRecall@10 = **0.223**、max = **0.951**、**50% catastrophic**——單一個位元就能抹掉幾乎全部 recall。

3. **全部 29 個 cell：0 crash、0 NaN/Inf。** 所有傷害都是 **silent-wrong（沉默式答錯）**——索引照常反序列化、照常回傳結果，只是結果錯了。這是可靠性上最危險的失效型態，因為現有的反序列化檢查完全攔不到。

**對核心問題的初步回答**：「量化是否更敏感」沒有單一答案，取決於正規化軸（見第 5 節）。在**每位元**層級，量化碼穩健、危險集中在共享元資料；在**每 MB** 層級，量化縮小的足跡反而把翻轉更集中地打在 recall 相關位元組上。淨效應留待 Phase 2 的 faults/MB 掃描決定。

---

## 2. 實驗設定

固定設計矩陣（取自 `qp/config.py`，所有相位凍結）：

| 參數 | 值 |
|---|---|
| IVF `nlist` | 1024 |
| 圖索引 `M` / `efConstruction` | 32 / 200 |
| 量測 `k` | 10 |
| 目標乾淨操作點 | recall@10 ≈ 0.95 |
| tolerant recall margin `ε` | 0.02 |
| 隨機種子 | 1234 |

**各索引調至同一乾淨操作點後的基線**（`../baseline.csv`）：

| 索引 | 量化 | 調節旋鈕 | clean recall@10 | 序列化大小 | 對齊 0.95？ |
|---|---|---|---:|---:|:--:|
| FLAT | — | 精確 | 1.000 | 512.0 MB | ✓ |
| IVF_FLAT | — | nprobe=21 | 0.95114 | 520.5 MB | ✓ |
| IVF_SQ8 | SQ8 | nprobe=24 | 0.95053 | 136.5 MB | ✓ |
| HNSW | — | efSearch=36 | 0.95171 | 784.1 MB | ✓ |
| HNSW_SQ8 | SQ8 | efSearch=40 | 0.95136 | 400.1 MB | ✓ |
| IVF_PQ_M8 | PQ | nprobe=96 | 0.37926 | 16.7 MB | ✗ |
| IVF_PQ_M16 | PQ | nprobe=128 | 0.563 | 24.7 MB | ✗ |

> **IVF_PQ 未納入單位元敏感度比較**：即使把 nprobe 開到最大，PQ 的乾淨 recall 也達不到 0.95（M8=0.379、M16=0.563），這是 PQ codebook 在不做 re-rank 下的天花板。由於本研究比的是「同一操作點下的敏感度」，PQ 對不齊就不能放進同一張比較表。它仍出現在第 3 節的區域結構與 Phase 2 規劃中。

**注入方法**：`faiss.serialize_index` → numpy `uint8` 陣列 → 對指定 (byte, bit) 做 XOR → `deserialize_index` → 查詢 → 還原該位元。大區域（vectors / codes）依語意位元位置**分層抽樣**；小區域（`sq_scale`）**全枚舉**。

> 本次 `sq_scale` 為全枚舉而非抽樣：該區域 1024 bytes = 256 個 float32 = 8192 bits，拆解為 256 sign + 2048 exponent + 5888 mantissa（high 3840 + low 2048）= 8192，與 `n_samples` 完全吻合。代表這份 vuln_map 是真實完整跑，非 smoke。

---

## 3. 各索引的位元組區域結構

不同索引把記憶體花在不同地方，這直接決定了「一個隨機位元翻轉會打到哪裡」。下表整理自 `regions_aug/*.regions.json`：

| 索引 | 總大小 | 主要區域組成 |
|---|---:|---|
| FLAT | 512.0 MB | header (45 B) + **vectors** fp32 (≈100%) |
| HNSW | 784.1 MB | graph_meta (1.5%) + **graph_edges** (33.2%) + **vectors** fp32 (65.3%) |
| HNSW_SQ8 | 400.1 MB | graph_meta + **graph_edges** (65%) + **sq_scale** 1 KB + **codes** uint8 (32%) |
| IVF_FLAT | 520.5 MB | header + **centroid** 512 KB (0.10%) + **codes** (99.9%) |
| IVF_SQ8 | 136.5 MB | header + **centroid** 512 KB (0.38%) + **sq_scale** 1 KB (0.0007%) + **codes** (99.6%) |
| IVF_PQ_M8 | 16.7 MB | header + centroid + **pq_codebook** 128 KB (0.79%) + codes (96%) |
| IVF_PQ_M16 | 24.7 MB | header + centroid + **pq_codebook** 128 KB (0.53%) + codes (97%) |

**高槓桿小區域**（單一翻轉可影響大量相依向量；取自 `../region_accounting.md`）：

| 索引 | 區域 | 大小 | 佔總量 |
|---|---|---:|---:|
| IVF_SQ8 / IVF_FLAT | centroid | 512 KB | 0.10–0.38% |
| HNSW_SQ8 / IVF_SQ8 | sq_scale | **1 KB** | **0.0003–0.0007%** |
| IVF_PQ | pq_codebook | 128 KB | 0.53–0.79% |

關鍵觀察：`sq_scale` 儲存的是「每維度的 vmin / vdiff 反量化參數」，被**整個索引的所有向量**共用。它只佔 0.0003% 的記憶體，卻是全域解碼的命脈——這正是它成為災難性目標的原因。

---

## 4. 單位元敏感度地圖（核心結果）

下表為 `vuln_map.csv` 全部 29 個 cell，依 ΔRecall@10 排序分組。`pct_benign`=|ΔRecall@10|≤1e-4 的比例；`pct_catastrophic`=ΔRecall@10>0.01（或 crash）的比例。

### 4.1 向量 / 碼區域 —— 全部穩健

| 索引 | 區域 | bit 位置 | n | mean ΔR@10 | max ΔR@10 | benign | silent-wrong |
|---|---|---|---:|---:|---:|---:|---:|
| FLAT | vectors | exponent | 249 | 4.8e-7 | 1.0e-5 | 100% | 12 |
| FLAT | vectors | mantissa-high | 463 | 2.2e-8 | 1.0e-5 | 100% | 1 |
| FLAT | vectors | mantissa-low | 250 | 0 | 0 | 100% | 0 |
| FLAT | vectors | sign | 38 | 2.6e-7 | 1.0e-5 | 100% | 1 |
| HNSW | vectors | exponent | 232 | 2.6e-7 | 2.0e-5 | 100% | 8 |
| HNSW | vectors | （其餘） | — | 0 | 0 | 100% | 0 |
| HNSW_SQ8 | codes | sq8_b0…b4 | 各125 | 0 | ≤1.6e-6 | 100% | 0 |
| HNSW_SQ8 | codes | sq8_b5 | 125 | 8.0e-8 | 1.0e-5 | 100% | 1 |
| HNSW_SQ8 | codes | sq8_b6 | 125 | 1.6e-7 | 1.0e-5 | 100% | 2 |
| HNSW_SQ8 | codes | sq8_b7 (MSB) | 125 | 8.8e-7 | 1.0e-5 | 100% | 11 |
| IVF_FLAT | codes | （全部 4 tag） | — | ≤4.4e-7 | ≤1.0e-5 | 100% | ≤11 |

**結論**：不論是 fp32 原始向量、還是 SQ8 量化碼，單一位元翻轉幾乎沒有影響。SQ8 codes 的最高位 `sq8_b7` 略有抬頭（11 個 silent-wrong）但 max 仍僅 1e-5。**量化碼沒有比全精度向量更脆弱——這是反直覺但清楚的結果。**

### 4.2 高槓桿元資料 —— 災難集中地

| 索引 | 區域 | bit 位置 | n | mean ΔR@10 | max ΔR@10 | benign | **catastrophic** | silent-wrong |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **HNSW_SQ8** | **sq_scale** | **exponent** | 2048 | **0.2230** | **0.9514** | 43.8% | **50.0%** | 1152 |
| **HNSW_SQ8** | **sq_scale** | **sign** | 256 | **0.0787** | **0.3746** | 50.0% | **50.0%** | 128 |
| HNSW_SQ8 | sq_scale | mantissa-high | 3840 | 0.00228 | 0.0970 | 78.4% | 6.6% | 1280 |
| HNSW_SQ8 | sq_scale | mantissa-low | 2048 | ≈0 | 0 | 100% | 0% | 0 |
| IVF_FLAT | centroid | exponent | 508 | 4.5e-4 | 0.00456 | 46.5% | 0% | 335 |
| IVF_FLAT | centroid | sign | 61 | 5.5e-5 | 7.3e-4 | 78.7% | 0% | 29 |
| IVF_FLAT | centroid | mantissa-high | 931 | ≈0 | 3.0e-5 | 100% | 0% | 19 |
| IVF_FLAT | centroid | mantissa-low | 500 | 0 | 0 | 100% | 0% | 0 |

**結論**：
- HNSW_SQ8 的 `sq_scale` 指數位元是整份地圖最危險的目標——半數翻轉造成災難性後果，最壞情況直接把 recall 從 0.951 砸到接近 0。因為 scale 是全域反量化參數，一動就讓**每個**向量的解碼座標整體位移。
- IVF_FLAT 的 `centroid` 指數位元造成大量 silent-wrong（508 次中 335 次），但幅度溫和（max 僅 0.0046）——一個 centroid 只影響它那條 inverted list 的向量，不是全域。

### 4.3 位元位置律

三類區域一致呈現：**exponent > sign > mantissa-high ≫ mantissa-low（≈0）**。完全符合 IEEE-754 浮點數值權重——翻動指數位元會把數值放大/縮小 2 的次方倍，翻動低位尾數只造成微不足道的擾動。這也說明為何 `sq_scale` 的 mantissa-low 即使全枚舉 2048 個位元，ΔRecall 仍是 0。

---

## 5. 「每位元」對「每 MB」的張力（連結 region_accounting 預測）

這是回答核心問題的關鍵——兩個正規化軸給出相反的直覺：

**(a) 每位元視角（本份單位元地圖）**：危險高度集中在共享元資料（`sq_scale`、`centroid`），而佔據絕大多數位元組的量化碼本身穩健。若按「翻一個特定位元」來看，量化沒有讓碼變脆弱。

**(b) 每 MB 視角（`../region_accounting.md`）**：硬體交付的是 per-bit 錯誤率，所以該問的是「一個隨機物理翻轉落在哪個 bucket」。各索引差異巨大：

| 索引 | recall_relevant | crash_structure（可偵測） | 解讀 |
|---|---:|---:|---|
| IVF_SQ8 | **100.0%** | 0% | 足跡極小、幾乎全是 codes，**每個**隨機翻轉都打在 recall 上 |
| HNSW_SQ8 | **32.0%** | **68.0%** | 68% 是圖結構（崩潰可偵測），形同「護盾」吸收翻轉 |

→ A4 預測：在相同 faults/MB 下，IVF_SQ8 與 HNSW_SQ8 用的是**同一個 scalar quantizer**，但 recall_relevant 足跡差 **68 個百分點**。這暗示「敏感度差異可能主要來自記憶體足跡 profile，而非量化器本身」。

**綜合結論**：量化會縮小索引（IVF_SQ8 僅 FLAT 的 27%、IVF_PQ 僅 3–5%），把同樣的物理錯誤率壓縮到更少、更密集且更高比例 recall 相關的位元組上。是「碼穩健」勝出、還是「足跡濃縮 + 脆弱共享參數」勝出，正是 Phase 2 faults/MB 掃描要拍板的淨效應。

---

## 6. 失效型態

跨全部 29 個 cell：

- **n_crash = 0**
- **n_nan_inf = 0**
- 其餘全是 **n_clean（benign）** 或 **n_silent_wrong**

意涵：在這個 corpus 上，位元翻轉**從不會**讓索引崩潰或吐出 NaN/Inf——它永遠安靜地反序列化、安靜地回傳一組「看起來正常但其實錯了」的鄰居。對可靠性而言這是最壞的情境：沒有任何例外、沒有任何哨兵值能在載入時攔截損毀。任何防護機制都必須是**主動的**（校驗碼 / 冗餘 / 重新驗證），不能指望被動的崩潰偵測。

---

## 7. 對 Phase 2 的建議

1. **把 faults/MB 掃描的火力集中在已定位的高槓桿區域**：`sq_scale`、`centroid`、`pq_codebook`。本份地圖已證明大區域（codes / vectors）近乎免疫，均勻掃描它們會浪費大量算力。

2. **保護性價比排序**：對 1 KB 級的 `sq_scale` 加校驗碼或三重冗餘，成本可忽略，卻能擋掉整份地圖最災難的失效模式（HNSW_SQ8 sq_scale exponent，mean ΔR 0.223）。這是最高 ROI 的防護點。

3. **加入 IVF_SQ8 vs HNSW_SQ8 的對照**作為 Phase 2 主軸，直接驗證 A4 的「足跡 profile 主導，而非量化器主導」假說（68 pp 預測差距）。

4. **連結 `../first-try/`**：first-try 發現「HNSW 圖結構損壞只影響速度、不影響 recall」。本份結果一致——HNSW/HNSW_SQ8 的 graph_edges 落在 crash_structure bucket，扮演吸收翻轉的護盾；而 recall 的命脈在 codes 與 scale。Q3 冗餘比較可沿此銜接。

---

## 附錄：資料來源

| 檔案 | 內容 |
|---|---|
| `vuln_map.csv` / `vuln_map.json` | 29 cell 單位元敏感度聚合表（本報告第 4 節主資料） |
| `regions_aug/*.regions.json` | 7 種索引位元組區域地圖（offset / len / dtype / semantic） |
| `../baseline.csv` | 各索引乾淨操作點、調節旋鈕、序列化大小 |
| `../region_accounting.md` | 記憶體 bucket 佔比與 A4 預測（第 5 節） |

*本報告僅分析既有數據，未修改任何實驗資料或程式碼。生成腳本：`phase1_sensitivity.py`（vuln_map）、`phase1_region_accounting.py`（regions_aug / region_accounting）、`phase0_build.py`（baseline）。*
