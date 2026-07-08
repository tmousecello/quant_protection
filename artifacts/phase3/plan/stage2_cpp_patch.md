# Claude Code 指示:Stage 2 — Option B(C++ EB-fallback 查詢路徑)

> 目標:讓實驗 B(ex 斜坡 + EB 買時間)可跑。作法 = patch C++ `exp_dumpids` 加 `--recovery {none|drop|fallback_eb}`,在 C++ 查詢期做 CRC 檢查與 EB-fallback(EB 排序需查詢期的 `est_dist`/`g_error`,Python 端不可跑——Option A 的包裝已證明會二次注入,已移除)。

---

## 0. 五條硬規則(違反會毀掉結果,先讀)

1. **注入只在 Python(`qp.faults`),C++ 絕不注入。** 腐蝕由 qp 對序列化 bytes 做、存成已腐蝕的 index 檔;C++ 只**載入並查詢**。這是注入的單一真理來源(Option A 二次注入教訓的一般化)。
2. **C++ 只吐 ids,recall 由 qp 算。** 沿用 Stage 0 parity 的 ids-dump 機制:per-query top-k ids 寫檔,Python 端用 `qp.metrics` 算 recall / `is_silent_collapse`。**不在 C++ 算 recall**——import-not-rewrite 跨語言同樣成立。
3. **EB-fallback 讀的必須是未腐蝕的 bin + factors。** EB 正確性依賴 bin/error-bound 沒壞(Stage 1 的依賴鏈)。實驗 B 只腐蝕 ex,自然成立,但:(a) 在 fallback 路徑的 code comment 標明這個假設;(b) 在輸出 metadata 記 `corrupted_regions=["ex_code"]`。之後做「bin 也腐蝕」的實驗時,這裡是第一個要改的地方。
4. **`--recovery` 三檔:`none | drop | fallback_eb`。** 科學問題是「EB 買多少時間」,對照組是 drop。`none` = 不檢查 CRC、直接用(可能壞的)ex;`drop` = CRC fail → 跳過該 candidate;`fallback_eb` = CRC fail → bin + error bound 悲觀排序。
5. **EB 語意 = Samuel exp 3 的既有機制,不發明新的。** 悲觀上界公式、error bound 用哪個欄位、tie-breaking,全部**從 Samuel repo 的既有 EB 實作讀出來對齊**。驗收 gate(§4)會抓語意分歧。

---

## 1. 開始前:讀這些
- Samuel repo:`exp_dumpids`(Stage 0 加的 ids-dump)、`exp_faultinject`、**exp 3 的 EB-fallback 實作**(悲觀排序的確切公式與欄位)。
- Stage 0:`qp/rabitq/layout.py`(ex chunk 的 offset/len、bin/factors 區)、adapter 的 ids 讀取端。
- E5 斜坡層的 Python 側(CRC chunk、counters)——本輪 C++ 端要與它的 chunk 定義**一致**(同 chunk 大小、同 CRC 演算法),否則 Python 記的 known-corrupted 和 C++ 判的對不上。
- 回報理解 + 實作 plan(patch 範圍、CRC 放哪、fallback 插在查詢路徑哪一步)。

---

## 2. 要建的東西

### 2a · C++ patch(`exp_dumpids` + `--recovery`)
- 載入(可能已腐蝕的)index 檔 + 一份 **CRC manifest**(見 2b)。
- 查詢路徑:rerank 取某 candidate 的 ex chunk 前驗 CRC:
  - `none`:不驗,直接用。
  - `drop`:fail → skip candidate。
  - `fallback_eb`:fail → 用 bin + error bound 悲觀排序(對齊 Samuel 語意);pass → 精確 ex 距離。
- 輸出:per-query top-k ids(ivecs 或既有格式)+ **stats json**(n_checked / n_crc_fail / n_fallback / n_dropped,per-query 與總計)+ provenance(platform/host/seed/index 檔 hash/`corrupted_regions`/recovery 檔位)。
- 決定性:同 index 檔 + 同 query + 同 flag → 相同 ids。

### 2b · CRC manifest(Python 產生,C++ 消費)
- Python 端(qp)在注入**前**對 clean index 的 ex chunks 算 CRC、寫 manifest(chunk 大小 config,預設對齊 E5 斜坡層的定義);注入後 index 檔 + manifest 一起交給 C++。
- 這保持「clean 的定義」在 qp 手上,C++ 只做比對。manifest 格式簡單(offset, len, crc)即可。

### 2c · Python 驅動(實驗 B runner)
- 迴圈:qp 產生腐蝕 index(E3c 的 4 種 pattern:uniform / clustered / cross_row / burst_accum,掃腐蝕比例)→ 呼叫 C++(三檔 recovery 各跑)→ 收 ids → `qp.metrics` 算 recall / collapse → 記錄(含 EB vs drop vs none 的 delta)。
- 輸出 jsonl(per-run 一列:pattern、fraction、recovery、recall@10、fallback 統計、provenance)。
- **provenance 一開始就蓋**(Stage 1 教訓):platform/host/seed/clean-baseline/index hash,不要事後補。


---

## 4. 驗收 gate(工作站,依序;過一關才進下一關)

1. **建置**:patch 後 9 binaries 編譯通過。
2. **No-op 錨**:clean index + `--recovery fallback_eb` → recall == 0.98376(無腐蝕時 EB 是 no-op;若不等,CRC 或 fallback 路徑在 clean 上誤觸發)。同時驗 `none`/`drop` 在 clean 上也 == 0.98376。
3. **Fig 2 錨(最重要)**:5% ex corrupted + `fallback_eb` → recall 落在 Samuel Fig 2 既有值(~0.94 plateau)附近(容差給 ±0.01 級)。**不過就停**:查悲觀上界公式 / error bound 欄位 / tie-breaking 的語意分歧,對齊後重跑。這一關過了,新 C++ EB 路徑才被證明與 exp 3 語意一致,sweep 的數字才可信。
4. **順序 sanity**:同一腐蝕 index,三檔的 recall 應呈 `fallback_eb ≥ drop`(Fig 1/2 的既有關係);高腐蝕端 `none` 應最差(沉默用壞資料)。
5. **決定性**:同 seed 重跑 → ids 相同。

全綠 → 跑 **4-pattern sweep**(實驗 B 主體)。

---

## 5. 給人類的 runbook(x86 工作站)
1. 套 patch、重建(`build_rabitq.sh`)→ gate 1。
2. 跑 gate 2–5(驅動附 `--gate` 模式,一鍵依序執行並回報)。
3. 全綠 → `--sweep`:4 patterns × 腐蝕比例掃描 × 3 recovery 檔,jsonl + log 帶回開發機。
4. **順手(便宜,實驗 A 收尾)**:
   - 無恢復基線加 per-tick recall 時間線(把 A 的「≤2 ticks」從推導變直接量測)。
   - (可與 sweep 並行)E3c 對 R=3 複本也注入(同 p、獨立流),實測多數決在複本同時累積下的表現——堵「複本不朽」那個 caveat。

---

## 6. 規則與邊界
- **Do**:對齊 Samuel EB 語意;chunk/CRC 與 E5 Python 側一致;三檔 recovery;ids-only 輸出;provenance 即時蓋;patch 小而保守;每步小提交。
- **Don't**:C++ 不注入、不算 recall;不發明新 EB 公式;不在 gate 3 未過時跑 sweep;不在開發機編譯真 C++;不造假;不手改產物。
- 不確定(尤其 EB 公式欄位、既有查詢路徑的插入點)→ report-and-stop。

---

## 7. 完成定義
- **你的交付(開發機)**:C++ patch + CRC manifest 工具 + 實驗 B 驅動(含 `--gate`/`--sweep` 模式)+ stub 測綠 + runbook。
- **科學驗收(人類,x86)**:gate 1–5 全綠 → 4-pattern sweep 產出 EB-fraction-vs-recall 曲線(實驗 B 主體,F3 斜坡半邊的資料)。

交付後回報「patch + 驅動就緒、stub 綠、runbook 就緒」;等工作站 gate 與 sweep 結果回來,再進分析與 Stage 2 第二輪。