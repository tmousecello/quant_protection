# Phase 2 實作指示

## 0. 最重要的執行約束（先讀，全程遵守）

- **只實作程式，不執行任何完整實驗或測量。** 你的工作是把下列腳本寫好、接上既有 `qp/` 函式庫與 `artifacts/`，讓它「能被人類一鍵跑」，然後停止。
- **允許且必須做的驗證**：每支腳本完成後，用極小子集做 smoke dry-run（秒級即可：少數幾個 bit 位置、查詢可只取數百筆），確認 pipeline 跑得通、輸出格式正確。**不要**跑全量。
- **禁止**：跑完整敏感度掃描 / rollup 全量 / burst 全掃。這些一律由人類執行。
- 交付時附一份 `artifacts/phase2/README.md`，寫清楚每支腳本「人類如何跑完整實驗」的指令、參數、預期輸出路徑與 schema、預期耗時，以及建議的執行順序。
- **不要改動** Phase 0/1 既有 artifacts，也不要覆寫 `qp/config.py` 既有的鎖定控制變因；需要新參數時用新增（附預設值），不要改舊值。
- 所有新輸出放在 `artifacts/phase2/` 底下，不要覆蓋 Phase 1 的 `artifacts/phase1/`。

---

## 0.5 乾淨起跑前置與本輪正確性修正（post-review，務必先讀）

Code review 後已修正一批正確性問題並重生區圖。**跑任何 Tier 3 / Tier 4 全量測量前**，必須先讓 Phase 1 在新幾何上重跑：

- **區域幾何變更**：`qp/regions.py` 現在把 IVF code 區塊前緣易崩潰的 invlist header 切成獨立 `codes_meta`（crash_structure），並把 `codes` 錨定到第一個真實 list code（fp32 grid 對齊）。`artifacts/regions/*` 與 `artifacts/phase1/regions_aug/*` **已重生**（含 PQ）。
- **`artifacts/phase1/vuln_map.csv` 為舊幾何且不完整**（缺 IVF_SQ8 與 PQ），**不可用於 Tier 3/4**。必須依序重跑：
  ```bash
  python phase0_build.py --regions-only     # 重建 artifacts/regions/*（已含 codes_meta）
  python phase1_region_accounting.py        # 散佈到 artifacts/phase1/regions_aug/*（快）
  python phase1_sensitivity.py              # 新幾何全量單位元掃描，含 IVF_SQ8（~6h，workstation）
  ```
- **Tier 3 Curve B 把 `graph_edges` 另列獨立曲線**：HNSW 與 HNSW_SQ8 共用同一份 graph_edges（受控變因、非編碼差異），其 ~2e9 bit 數會淹沒量化訊號並讓 fp32 看似崩潰；主曲線排除它後 fp32≈0 成立（見 §6）。
- **失效計數**：harness 例外（子行程 import 失敗等）標 `harness-error` 並讓掃描大聲中止，**不計入** corruption `crash`。
- **可重現**：`--validate-multiflip` 改用穩定 stream id（非 `hash(name)`）。

---

## 1. 背景與現況（精簡）

研究問題：記憶體位元翻轉（DRAM soft error / 儲存損毀）下，量化向量索引的 recall 是否比全精度更脆弱。

Phase 0/1 已確立：大宗 codes/vectors 對單位元翻轉**幾乎免疫**；危險集中在 KB 級的共享 metadata（SQ8 的 `sq_scale`、IVF 的 `centroid`）；最嚴重為 `HNSW_SQ8 / sq_scale` 指數位元（mean ΔR@10≈0.223、max 0.951、50% catastrophic）；全部 **silent-wrong，0 crash、0 NaN/Inf**；位元位置律 exponent > sign > mantissa-high ≫ mantissa-low(≈0)。

既有資產（直接沿用，勿重造）：
- `qp/config.py`（鎖定矩陣：IVF nlist=1024、圖 M=32/efC=200、k=10、tolerant ε=0.02、seed=1234）
- `qp/data.py`、`qp/metrics.py`（recall@{1,10,100} + tolerant recall + failure-mode 分類器 silent/nan_inf/crash）
- `qp/flip.py`（serialize→XOR→deserialize→還原 substrate）
- `qp/regions.py`、`artifacts/regions_aug/*.regions.json`（位元組區域地圖）
- `artifacts/baseline.csv`（各索引乾淨操作點、旋鈕、序列化大小）
- `artifacts/phase1/vuln_map.csv`（Phase 1 單位元地圖，新輸出沿用其 schema）
- 既有腳本可參考：`phase1_sensitivity.py`、`phase1_region_accounting.py`、`phase0_build.py`

---

## 2. 本輪已鎖定的設計決定（勿重新討論）

1. **PQ 用自身基線 + collapse 機率**。本輪只做 **IVF_PQ_M8 / IVF_PQ_M16**（皆已建好）；HNSW_PQ 留 Phase 3。PQ 不需對齊 0.95，直接用 `baseline.csv` 內各 PQ 索引自己的 clean recall（M8≈0.379、M16≈0.563）當基準。
2. **主軸論述 = 「量化引入了 fp32 沒有的災難性單點結構」**：fp32 索引（FLAT/IVF_FLAT/HNSW）無單位元災難目標；SQ8 多出 `sq_scale`、PQ 多出 `pq_codebook`。`IVF_SQ8 vs HNSW_SQ8` 對比降為附錄（期望 ΔRecall，由 centroid 主導）。
3. **偵測用 range-check（值域檢查），不是 NaN/Inf**（因為災難是 silent-finite-wrong）。NaN/Inf 僅作便宜附加項。
4. **burst/clustered 為獨立子研究**，本輪做 **IVF_PQ_M8 / HNSW_SQ8 / IVF_FLAT** 三個。
5. **正規化用絕對「每位元」錯誤率**（物理上有意義），collapse 風險由臨界結構的**絕對 bit 數**驅動，不用「每次翻轉的條件機率」。

---

## 3. 共用實作規格

- 沿用 `qp/flip.py` 的注入/還原；沿用 `qp/metrics.py` 的失效分類（silent / nan_inf / crash）。
- 輸出 schema 與 Phase 1 `vuln_map.csv` 一致（可新增向後相容欄位）：至少含 `index, region, bit_position_tag, n, mean_dR@10, p99_dR@10, max_dR@10, pct_benign, pct_catastrophic, n_silent_wrong, n_crash, n_nan_inf, region_bits, baseline_mode`。新增 `baseline_mode` 欄位標記 `own`（PQ）或 `aligned`（其餘）。
- 種子固定、可重現；所有抽樣參數走 CLI 旗標或新 config 區塊，附合理預設。
- **catastrophic（collapse）定義**走 config 參數，預設：對齊基線索引用「ΔR@10 > 0.01 絕對」；PQ（低基線）用「recall retention < 50% of own clean」相對定義。兩者都記錄，供 Tier 3 取用。
- **崩潰隔離（重要）**：graph_edges 與 burst 注入可能讓 FAISS C++ 直接 segfault 整個行程。對這兩類注入，**每次翻轉在獨立 subprocess 內執行**（或等效隔離），使 segfault 被記為 `crash` 而不會殺掉整個掃描。Tier 1（codes/codebook/centroid）依 Phase 0 dry-run 為乾淨反序列化，可不需隔離。

---

## 4. Tier 1 — PQ 單位元敏感度地圖（腳本 `phase2_pq_sensitivity.py`）

目的：測 PQ 的碼是否打破「碼免疫」（PQ 碼是 codebook 索引、翻轉是離散跳躍），並把 `pq_codebook` 從推測變實測。

步驟：
- [ ] 載入 `IVF_PQ_M8`、`IVF_PQ_M16` 索引、其 `regions_aug` 與 `baseline.csv` 的 own clean recall。
- [ ] 從 region 地圖取 `recall_relevant` 區域：`pq_codes`（uint8 索引）、`pq_codebook`（fp32 子量化中心）、`centroid`（fp32 粗量化中心）。
- [ ] bit 位置標籤：
  - `pq_codes`：依 byte 內 bit index 0..7 標 `code_b0..code_b7`（註明這是**索引位元、非 IEEE-754**，無 exponent/mantissa 語意）。
  - `pq_codebook`、`centroid`：fp32 → 標 `sign / exponent / mantissa-high / mantissa-low`（沿用 Phase 1 對 fp32 的拆法）。
- [ ] 抽樣策略：`pq_codes` 大宗區域依 Phase 1 的 S 抽樣；`pq_codebook`（≈128 KB ≈ 1.05M bits，全枚舉太貴）**預設依 bit 位置分層抽樣**（樣本數足以解析 p99/尾巴），另留 `--full-enum-codebook` 選項。`centroid` 沿用 Phase 1 設定。
- [ ] 對每個抽到的 bit：`qp/flip` 翻轉 → 用固定查詢重跑 → 算 recall@{1,10,100}+tolerant → ΔRecall 相對**該 PQ 索引自身** clean → 失效分類 → 還原。
- [ ] 依 (index × region × bit_position_tag) 聚合（mean/p99/max ΔR、%benign、%catastrophic 用相對定義、failure 計數、region_bits、baseline_mode=own）。
- [ ] 輸出 `artifacts/phase2/vuln_map_pq.csv`（同 schema）。
- [ ] **smoke**：每區域取數個 bit、查詢取數百筆，跑通驗證輸出 → 停。

---

## 5. Tier 2 — graph_edges 注入特性化（腳本 `phase2_graph_sensitivity.py`）

目的：釘死「圖足跡是海綿」這個前提——graph 邊翻轉到底是 crash（可偵測）／silent 但對 recall 無害（first-try 的只影響速度）／silent 有害（誤導路由）。

步驟：
- [ ] 載入 `HNSW`、`HNSW_SQ8` 與 `regions_aug`；定位 `graph_edges`（與 `graph_meta`）區域。
- [ ] bit 位置標籤：邊是節點 id（int32）。標 `id_high`（高位，預期 → 越界 id → crash）vs `id_low`（低位 → 合法但錯鄰居 → silent）；或記錄完整 bit index。
- [ ] **以 subprocess 隔離**每次翻轉（見 §3）：翻轉 → 還原索引 → 重跑查詢 → 分類 `crash` / `silent-benign`（ΔR≈0，可選記 QPS 變化）/ `silent-harmful`（ΔR>門檻）→ 還原。
- [ ] 抽樣 graph_edges（區域大，抽樣即可），每索引輸出三分類比例（%crash / %silent-benign / %silent-harmful）、ΔRecall 分布、可選 QPS delta。
- [ ] 輸出 `artifacts/phase2/graph_edges_characterization.csv`。
- [ ] **smoke**：少量邊 + 小查詢集，驗證 subprocess 隔離能正確捕捉 crash → 停。

---

## 6. Tier 3 — faults/MB rollup（腳本 `phase2_rollup.py`，純分析、不翻 bit）

目的：合成兩條曲線，回答主問題。輸入全部來自既有地圖，**不做新的注入**（多翻轉驗證為可選小程序）。

步驟：
- [ ] 載入並合併：`artifacts/phase1/vuln_map.csv` + `vuln_map_pq.csv`（Tier 1）+ `graph_edges_characterization.csv`（Tier 2）+ 區域大小（`regions_aug` / `region_accounting`）。
- [ ] 對「每位元錯誤率」`r` 做掃描（預設範圍 `r ∈ {1e-9 … 1e-5}`，並在 README 註明可換算自 DRAM field-study 速率，如 Schroeder 的 FIT/Mbit；範圍走 config）。
- [ ] 計算曲線 A「期望 ΔRecall(r)」= Σ_region E(該區域翻轉數 @ r) × E(ΔRecall│單翻轉)。大宗 ≈0、centroid 溫和、metadata 罕見大值，線性加權近似即可。**僅在 `region_bits·r ≪ 1` 成立；超出時把總和 clamp 到物理範圍 [-1,1]（並丟棄非有限項），避免出現 >1 的不可能 recall 下降。**
- [ ] 計算曲線 B「P(catastrophic collapse)(r)」= 1 − Π_region P(該區域無致命翻轉)，致命翻轉率 = 該區域 catastrophic 比例 × region_bits × r。**由 `sq_scale`（SQ8）與 `pq_codebook`（PQ）絕對 bit 數驅動；fp32 索引 ≈ 0。**
- [ ] **`graph_edges` 另列獨立曲線、排除於主曲線之外**：HNSW/HNSW_SQ8 共用同一份 graph_edges（受控變因），其巨大 bit 數會淹沒 sq_scale 量化訊號並讓 fp32 看似崩潰。主曲線只含 `series=main`；graph 輸出到 `curveB_graph_edges.csv`。（邊翻轉的 crash 可偵測，故 graph 的 cat_frac 只用 silent-harmful。）
- [ ] **主輸出**：曲線 B by index（fp32≈0 vs SQ8 有 sq_scale 項 vs PQ 有 codebook 項，**不含 graph_edges**）——即「量化引入 vs fp32 沒有」主軸圖的資料。
- [ ] **附錄輸出**：曲線 A 的 `IVF_SQ8 vs HNSW_SQ8`（centroid 主導）；以及 graph_edges 的獨立 A/B 曲線。
- [ ] 正規化用**絕對每位元率**（README 寫明：collapse 由結構絕對大小驅動，非每次翻轉條件機率）。
- [ ] 可選 `--validate-multiflip`：對 `sq_scale` / `pq_codebook` 同時翻 k 個 bit（k=2..數個）做少量實證，驗證非可加性假設（種子穩定可重現）。
- [ ] 輸出 `artifacts/phase2/rollup/curveA_expected_dR.csv`、`curveB_collapse_prob.csv`、`curveB_graph_edges.csv`、`curveA_graph_edges.csv`、`region_terms.csv`（含 `series` 欄）。
- [ ] **smoke**：以既有地圖跑一次、確認兩曲線產出 → 停（本 tier 本來就快）。

---

## 7. Tier 4 — range-check 偵測護欄（腳本 `phase2_detection.py`）

目的：落實「偵測用值域檢查」的修正，量它能擋掉多少災難。

步驟：
- [ ] 從**乾淨**索引算各 metadata 區域（`sq_scale` / `centroid` / `pq_codebook`）的訓練值域（min/max，或穩健分位界），存 `artifacts/phase2/detection/expected_ranges.json`。
- [ ] 實作 guard：給定（可能受損的）metadata，檢查所有值是否落在預期值域（+ NaN/Inf 便宜附加檢查），回傳是否偵測到、命中哪個區域。
- [ ] 評估：把 Tier 1 / Phase 1 產生的受損實例（尤其 catastrophic 者）餵進 guard，量：對 catastrophic 翻轉的涵蓋率、對 moderate 翻轉的涵蓋率、誤報率（對乾淨與 benign 翻轉跑 guard）、開銷（掃 metadata 時間 vs 一次查詢時間）。
- [ ] 輸出 `artifacts/phase2/detection/coverage.csv`。
- [ ] **smoke**：少量受損實例驗證 guard 回傳與計數正確 → 停。

---

## 8. 獨立子研究 — burst / clustered 注入（`qp/flip.py` 擴充 + 腳本 `phase2_burst.py`）

目的：測空間相關錯誤（壞 DIMM 區塊）。預期 burst 暴露度隨臨界結構大小放大：PQ codebook(128KB) ≫ SQ sq_scale(1KB) ≫ fp32(無)。

步驟：
- [ ] 在 `qp/flip.py` 新增 `burst_flip(buf, start_bit, B)`：對連續 B 個 bit 做 XOR，附對應還原；**不改既有單 bit 介面**。
- [ ] `phase2_burst.py` 對 `IVF_PQ_M8` / `HNSW_SQ8` / `IVF_FLAT`：
  - 掃 burst 長度 `B ∈ {1, 8, 64, 512, 1024, 8192, 65536, 262144} bits`（涵蓋 sq_scale 8K bits 到 codebook 1M bits 尺度；走 config 可調）。
  - 每個 B：N 次試驗，起點在該索引 byte 範圍內**均勻隨機**；burst 翻轉 → 重跑查詢 → 量 ΔRecall + collapse + 失效型態 + 命中了哪個區域 → 還原。
  - 可選 `--aligned-worstcase`：起點對齊 `sq_scale` / `pq_codebook` 起點。
  - **以 subprocess 隔離**（burst 可能打進 header/graph 而 crash）。
- [ ] 輸出 `artifacts/phase2/burst/<index>_burst.csv`（每 index×B：P(collapse)、ΔRecall 分布、各區域命中比例、failure 計數）。
- [ ] **smoke**：小 B、少量試驗，驗證 burst_flip 還原正確、subprocess 捕捉 crash → 停。

---

## 9. 交付與停止條件（Acceptance — 全部滿足才算完成）

- [ ] 新腳本就位：`phase2_pq_sensitivity.py`、`phase2_graph_sensitivity.py`、`phase2_rollup.py`、`phase2_detection.py`、`phase2_burst.py`；`qp/flip.py` 已加 `burst_flip`。
- [ ] 全部可乾淨 import（`python -c "import ..."` 或 `--help` 正常）。
- [ ] 每支腳本通過一次 smoke dry-run（小子集、秒級），並在 README 記下 smoke 指令與「已跑通」狀態。
- [ ] `artifacts/phase2/README.md` 寫齊：每支腳本的**完整實驗指令**、參數、輸出路徑與 schema、預期耗時/成本、建議人類執行順序（Tier 1 與 Tier 2 可並行 → Tier 3 rollup → Tier 4 → burst 獨立）。
- [ ] **未執行任何完整測量**；未改動 Phase 0/1 artifacts 與 `qp/config.py` 鎖定值。
- [ ] 最後回報「建了什麼 + 完整實驗該怎麼跑」的摘要，然後**停止**。