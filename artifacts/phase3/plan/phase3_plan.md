# Phase 3 執行計畫（依賴排序版）

> **關鍵路徑（critical path）**：D1 共享介面 → bug ①② + harness→RaBitQ → **E1 RaBitQ 刻畫** → E5 scrub 機制 + E3c temporal → **E4 收斂 / F3**。其餘（E2、E3a/b、Milvus Tier0、bug ③④）都在關鍵路徑之外、可並行或延後。

---

## Stage 0 — 前置（這些沒完成，後面都不要動）

### 決定（先鎖）
- **D1 · 共享介面（最關鍵，否則 Bottom-Up / Top-Down 不會會合）**：
  - 目標 recall：畫成 frontier **R ∈ {0.85, 0.9, 0.95}**，0.9 為 headline 點（不要只押 0.9）。
  - fault model：advisor 的三個（uniform p、spatial (W,k)、temporal burst）。
  - index：RaBitQ。
  - 成本度量：記憶體 bytes + **scrub 開銷/頻率**。
  - recall 度量：recall@10 + tolerant recall。
- **D2 · 分工**：A/B/C/D/E 工作流負責人（團隊內部，快速定）。

### 修復（blocker，依序）
- **bug ①**：`is_silent_collapse` 一併排除 `NAN_INF`（單一真理來源）。**先修**。
- **bug ②**：退役 recompute 捷徑，改用原生全量 `pct_collapse`；只對 harmful>0 的區域實測 collapse。**必須在 ① 之後**（否則 fp32 矛盾復發）。
- > bug ③④ 不在本階段——它們需要伺服器 + raw 記錄，排到 Stage 4，不擋 sprint。

### 設置
- 把既有注入 harness 延伸到 **RaBitQ**（對齊 Milvus IVF_RABITQ 佈局：rotation / factors / bin / ex / pointer）。
- **查證旋轉矩陣佈局**：稠密 d×d（SIFT 64 KB）還是結構化快速旋轉（更小）——決定區域地圖與足跡。（這是 fact-find，不是決定。）

**Gate 0**：D1 已鎖 + bug ①② 修好 + RaBitQ harness 就緒 + 旋轉佈局已知 → 才可進 Stage 1 的 E1。

---

## Stage 1 — 刻畫與原語（四項可並行；E1 在關鍵路徑上）

### 實驗
- **E1 · RaBitQ 脆弱度刻畫【NEW，critical】**（依賴 Gate 0）
  - 用修正後 collapse 定義刻畫 rotation / factors / bin / ex / pointer，按 bit 位置。
  - **產出**：F1（風險局部化，預期旋轉矩陣＝RaBitQ 的單點災難）+ **Top-Down 的削減順序**（criticality 排序）+ 保護/scrub 分配。
- **E2 · Bottom-Up 容忍曲線【DONE，需收尾】**（依賴 D1）
  - Samuel 既有 EB-fallback 圖：加 **p 副軸**（f=1−(1−p)⁸³²）、標出**跌破 0.9 的腐蝕上限**（約 ex ~10%、p ~1e-4）。
  - **產出**：F2 + 餵 Top-Down 的「原語容忍上限」。
- **E3a · model 1 可加性驗證【NEW，小】**：小臨界結構多翻轉是否疊加（補 rollup 嚴謹度）。
- **E3b · model 2 spatial (W,k)【NEW，小】**：把 burst 細化成 window 內稀疏聚集 + 相鄰-row 型。

**Gate 1**：E1 給出「要保護/scrub 哪些結構 + 削減順序」、E2 給出「原語容忍上限」、E3a/b 給出靜態 fault-model 涵蓋 → 才能組裝機制與時間維度。

---

## Stage 2 — 機制與時間維度（依賴 E1）

### 實驗
- **E5 · 三層 scrub-repair 機制【NEW，critical】**（依賴 E1 的結構清單）
  - 實作：小臨界結構複製 + 週期 scrub-repair；ex 走 CRC + EB-fallback + 懶 reload；圖 bounds-check+skip；大宗不 scrub。
- **E3c · model 3 temporal timeline【NEW，critical】**（依賴 E5 的 scrub 概念）
  - 模擬時間軸：查詢與錯誤事件交錯、腐蝕累積、可選觸發 burst；量 recall over time。
  - **產出**：F3 所需的「累積動態」。

**Gate 2**：機制可跑 + 有了腐蝕隨時間累積的動態 → 才能做收斂的 scrub-interval frontier。

---

## Stage 3 — 收斂（Bottom-Up × Top-Down 會合點）

### 決定（做 F3 前鎖）
- **D4 · F3 的 baseline 集**（advisor 證據門檻意見收進來）：至少 **uniform 複製 + 立即 reload** 兩條參考。

### 實驗
- **E4 · Top-Down 成本最小化 → F3【NEW，critical，收斂】**（依賴 E1 削減順序 + E2 容忍上限 + E3c 累積 + E5 機制）
  - 在 recall ≥ R 約束下，用 E1 的順序削成本到最小；scrub 間隔調到「把 ex 累積壓在 E2 的容忍上限內」。
  - **產出**：**F3 scrub-interval frontier**（三層政策 vs uniform/reload，EB fallback 延長安全間隔）。

**Gate 3**：F1 + F2 + F3 三張 money figure 在手 = 論文證據骨架完成。

---

## Stage 4 — 完整性與投稿（多數在伺服器；寫作全程並行）

### 修復（伺服器）
- **bug ③**：重跑 `phase2_detection.py` 補 collapse 區塊（產物一律碼生成）。
- **bug ④**：以原生全量 10k-query 為單一來源，**重生最終 cat_frac/cat_bits**。

### 其他
- **Milvus Tier 0**：把生產缺口（記憶體服務無逐查詢檢查、恢復＝慢速重載、IVF_RABITQ 原生對應）寫進 motivation。
- **決定 D5 · hook 定稿**（可更早；寫作時鎖死）。
- **寫作**：framing/intro 先行（全程並行）→ 完整初稿 → 打磨 → 內部 + advisor 審 → 定稿。

---

## 一頁速查

### 決定清單（依序）
| # | 決定 | 階段 | 阻塞什麼 |
|---|---|---|---|
| D1 | 共享介面（recall frontier / fault models / index / 成本 / recall 度量） | 0 | Bottom-Up×Top-Down 會合、E1/E2/E4 全部 |
| D2 | 工作流分工 | 0 | 執行 |
| D4 | F3 baseline 集 | 3 | F3 定稿 |
| D5 | hook 定稿 | 4（可更早） | 寫作 framing |

### 實驗清單（依序，標新舊）
| 順序 | 實驗 | 狀態 | 依賴 | 產出 |
|---|---|---|---|---|
| 1 | harness→RaBitQ + 旋轉佈局查證 | NEW（設置） | bug ①② | E1 的前置 |
| 2 | **E1 RaBitQ 刻畫** | **NEW** | Gate 0 | F1 + 削減順序 + 分配 |
| 2∥ | E2 容忍曲線收尾 | DONE→收尾 | D1 | F2 + 容忍上限 |
| 2∥ | E3a 可加性驗證 | NEW（小） | bug 修 | rollup 嚴謹度 |
| 2∥ | E3b spatial (W,k) | NEW（小） | burst harness | model 2 涵蓋 |
| 3 | **E5 三層 scrub 機制** | **NEW** | E1 | 機制 |
| 3 | **E3c temporal timeline** | **NEW** | E5 | 累積動態 |
| 4 | **E4 Top-Down → F3** | **NEW** | E1+E2+E3c+E5 | F3（headline） |
| 5 | bug ③④ 重生 / Milvus Tier0 | 收尾 | 伺服器 | 最終數字 + motivation |

---

## 給團隊的一句話
**先把 Stage 0 清乾淨（鎖 D1、修 bug ①②、接好 RaBitQ harness），其餘卡點就都開了。** 之後關鍵路徑是 E1 → E5+E3c → E4，E2 與兩個小 fault model 並行不擋路，bug ③④ 與 Milvus 留到最後在伺服器收。三張 money figure（F1 局部化、F2 容忍、F3 scrub-interval）一到手，就轉全力寫作。