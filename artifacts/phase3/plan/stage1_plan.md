# Claude Code 指示:Phase 3 — Stage 1(E1 刻畫 + E3a/E3b 小實驗)

> 這份是給你(Claude Code)的自含 brief。**請先讀「工作流」與「開始前」兩節**——這個 stage 的工作方式和一般不同:你在開發機上開發,但**真正的 RaBitQ 跑由人類在 x86 工作站執行**(那台不方便登入 Claude/GitHub)。你的產出是「對 stub 測過、可直接搬上工作站跑」的程式 + runbook。

---

## 工作流(最重要,先讀)
- **你(Claude Code)只在開發機(arm64)工作。RaBitQ 在 arm64 編不過,所以你不會、也不要嘗試跑真 RaBitQ。**
- 你對著一個 **stub adapter**(確定性假 adapter,介面與 Stage 0 的真 adapter 相同)開發與單元測試,確認整條 pipeline 邏輯正確、輸出格式對。
- runner 必須 **adapter 注入 + config 驅動**:同一支程式,靠 flag/config 切換 stub(開發機)或真 adapter(工作站),**工作站上不需改任何碼**。
- runner 要輸出**豐富 log + per-flip 原始 jsonl + 中間狀態**,這樣萬一工作站上(沒有 Claude)出問題,人類能把 log 帶回開發機配你除錯。
- 你的交付 = stub 測試全綠 + config + 一份 **runbook**(人類照著在工作站跑)。**真正的科學結果(vuln_map / F1)由人類在工作站產生**,不是你。
- **report-and-stop**:不確定的東西(如某結構的 bit-class 分類)回報,不要猜後硬幹;不要假造任何數字。

---

## 開始前:先讀與探索(先做,再提 plan)
1. 讀根目錄 `CLAUDE.md`、Stage 0 的 status(已驗收:42/42、baseline 0.98376)。
2. 探索 Stage 0 在 `main` 上建好的東西,理解你要對接/注入 stub 的介面:
   - `quant_protection/qp/rabitq/layout.py`、`adapter.py`:真 adapter 的介面(region map、`query_ids`、serialize/deserialize、`clean_baseline_recall`)。stub 要鏡像這個介面。
   - `qp/faults.py`(`uniform_p` / `spatial_cluster` / `cross_row` / `temporal_burst` / `CumulativeCorruption`)、`qp/bits.py`、`qp/metrics.py`(`is_silent_collapse`、recall、failure 分類)。
   - `qp/rabitq/layout.py` 的 `serialized_region_map`:**每個結構的 bit-class 分類請從 layout 推導**(見下「bit-class」一節),不要假設全是 float32。
3. 回報你的理解 + Stage 1 的實作 plan。

---

## 背景(精簡,讓你理解為什麼)
- **主題**:向量索引在會復發的記憶體位元錯誤下的資料保護。容錯風險高度不均——危險集中在 KB 級全域共享解碼結構。
- **Stage 1 在做什麼**:把 Phase 1/2 對 FAISS 做過的脆弱度刻畫,搬到 **RaBitQ**。產出 **F1(風險局部化)+ Top-Down 的削減順序(criticality 排序)+ 三層保護/scrub 分配**。
- **headline 預期**:Stage 0 從源碼確認,RaBitQ 的全域旋轉是 `FhtKacRotator.flip_`——一個**結構化 FHT+Kac 的 sign-flip 位元向量,SIFT 上只有 64 bytes / 512 bits**,寫在檔尾、被所有向量共享。預期它是 RaBitQ 版的 `sq_scale`(單點災難)。機制:corrupt 它 → query 端旋轉與 database 編碼基底**不匹配** → 所有距離估計錯 → 預期**沉默崩潰**(finite-but-wrong,非 nan-inf)。但**這是 E1 要實測的,不是已知**——結構化/正交也可能比 sq_scale 更 graceful;別在量到之前當定論。
- **collapse 定義(修正後)**:`is_silent_collapse` = 保留率 < 50% of own clean recall@10,**只算沉默**(排除 crash 與 nan-inf)。用 `qp.metrics` 的版本。

---

## 核心規則(違反會毀掉結果)
1. **量測/度量/注入一律 import 自 `qp/`,不重寫。** `is_silent_collapse`、recall、fault 注入器是同一份碼。
2. **用原生全量 `pct_collapse`,不要用退役的 `phase2_recompute_collapse` 捷徑。** 那支是 SIFT/SQ8 專用、硬編碼「只掃 sq_scale、其餘強制 0」,套到 RaBitQ 的新區域(rotation/factors/bin/ex/pointer)會誤拋或把真訊號歸零。對每個 harmful>0 的區域**實測** collapse。
3. **bit-class 分類從 layout 推導,不要假設 float32。**(見下節)
4. **stub 開發、真跑交人類。** 你驗證的是「pipeline 跑得通、注入打在對的 region、輸出格式對」,不是真 recall 數字。
5. report-and-stop;不造假;產物一律碼生成。

---

## bit-class 分類(每個結構不同,從 layout 推導)
- **rotation(`flip_`,sign-flip 位元向量)**:bit = sign-flip 位元,**無 exponent/mantissa**。512 bits 可**完整列舉**。若 rotator 有可辨識的子階段(Hadamard/Kac stage),讀源碼後可分組;否則當 512 個平坦 bit。
- **factors(`bin_factors` 12B / `ex_factors` 8B)**:多半是 float(scaling、**error bound**)。套 float bit-class(exponent/sign/mantissa)。**error bound(f_error)特別標記**——它被恢復機制用,corrupt 它影響 EB fallback。讀 layout 確認有幾個 float、各自角色。
- **bin(1-bit codes)**:每維 sign bit。bit = 哪一維的 sign。
- **ex(擴展位)**:量化值的低位 bit。bit-class = 高位 vs 低位擴展 bit(類比 mantissa-high/low)。
- **pointer(graph links)**:整數索引。bit-class = 高位 vs 低位 index bit(高位 → 大跳躍 → 可能 crash/越界;低位 → 小擾動)。

不確定某結構的正確分類就**回報**,不要硬分。

---

## 要建的東西

### 6.0 · Stub adapter(開發機測試用)
鏡像真 adapter 介面(region map、`query_ids`、serialize/deserialize、`clean_baseline_recall`),回傳**確定性**的假結果。用途:讓你在無 RaBitQ 的開發機上把下面 runner 的邏輯測到綠。**不需真 recall 行為**,只需介面一致 + 輸出確定。

### 6.1 · E1 RaBitQ 脆弱度刻畫 runner【critical】
- adapter 注入 + config 驅動。對每個結構(rotation / factors / bin / ex / pointer)× 每個 bit-class:
  - 注入**單一位元翻轉**(主刻畫),量 `recall@10`、`is_silent_collapse`、failure 分類(silent / nan-inf / crash)。
  - **rotation**:完整列舉 512 bits(全域、小,像 sq_scale 那樣窮舉)。
  - **per-vector 結構(factors/bin/ex/pointer)**:抽樣 N 個向量 × bit 位置,帶 bootstrap CI。
- 聚合成 **vuln_map**:每 結構×bit-class 的 `pct_collapse / pct_harmful / pct_crash / pct_nan_inf`(+ CI)。
- 由 vuln_map 導出:**criticality 排序(Top-Down 削減順序)** + **三層 scrub 分配**(小臨界 rotation/bin/factors → 頻繁 scrub;ex → CRC+EB+懶 reload;免疫 → 不 scrub;pointer → bounds-check)。

### 6.2 · E3a 可加性驗證【小】
- 用單位元 map 依 rollup 模型(`P_collapse ≈ 1−(1−r)^cat_bits` / 線性疊加)**預測**小臨界結構(rotation、factors)的多位元 collapse。
- 再對同結構**實際注入多位元**(model 1 uniform 的 k-flip),比對「實測 vs 預測」。
- 產出:可加性成立/失效的區間(失效=飽和或複合效應)。

### 6.3 · E3b spatial (W,k)【小】
- 用 Stage 0 的 `spatial_cluster(W,k)` + `cross_row`,對結構注入**window 內稀疏聚集**與相鄰-row 型。
- 對比「相同 bit 預算、不同空間分布(聚集 vs uniform)」的 collapse/recall;驗證 Phase 2 的「暴露度 ∝ 1/分散度」在 RaBitQ 上是否成立(聚集打在小結構應更危險)。

### 6.4 · 輸出
- `vuln_map`(F1 的資料)、criticality 排序、scrub 分配、E3a/E3b 結果。
- **per-flip 原始 jsonl + 豐富 log**(供工作站失敗時離線除錯)。三支都寫:E1 `raw/rabitq.records.jsonl`、E3a `raw/e3a.records.jsonl`、E3b `raw/e3b.records.jsonl`(共用 `qp.rawio.RawWriter`)。
- **runbook**(見下)。

> **蓋章後記(review 缺陷 #1–#3 修正)**:每個結果 JSON 都帶頂層 `meta`(`qp.provenance`):`units` 圖例(`pct_*`=percent 0-100、`collapse_frac`/`p1`/`frac_*`=fraction 0-1,解掉 0.781% 被讀成 78% 的歧義);`platform_confirmed_real`=**確認非推定**(real ∧ x86_64 ∧ clean 在 plateau 才 True;arm64/stub 為 False);另含 clean_baseline、seed、index_geometry、RaBitQ commit、dep 版本。criticality collapse 改為 `pct_collapse_worst_bucket`(驅動排序)+ `pct_collapse_overall`(全 bit n-加權)+ 各自 `frac_*` 孿生。**最終蓋章產物需在 x86 重跑、`platform_confirmed_real:true` 才算數。**

---

## Stub 測試(開發機驗收 = 你的交付門檻)
對 stub assert:
- pipeline 端到端跑通,輸出 schema 正確(vuln_map / 排序 / 分配 / jsonl)。
- 注入**確實打在 region map 指定的 bytes**(用 stub 的已知 region 驗:注入「rotation」region 只動到 rotation bytes、restore 後位元組相同)。
- recall/collapse 走的是 import 的 `qp.metrics`、`is_silent_collapse`(含 nan-inf=False)。
- config 切換 stub↔真 adapter 不需改碼。
- 決定性(同 seed 同結果)、CI 計算正確、E3a 的預測 vs 實測比對邏輯正確、E3b 的聚集分布正確。

---

## 給人類的 runbook(工作站執行)
1. 同步程式上工作站(rsync/deploy-key),設 config 用**真 adapter**。
2. **第一發:擾動 rotation region 的單一 bit,看 recall。** 這同時 (a) 物理確認 region map 切對、(b) 是 E1 第一個資料點(回答「64B 旋轉是不是災難性」)。若 recall 沒崩,先停、回報——可能是 map 偏移或旋轉比預期 graceful,兩者都要知道。
3. 跑完整 E1(全結構 × bit-class)、E3a、E3b。開 verbose log、存 per-flip jsonl。
4. 把 vuln_map / 排序 / 分配 / log 帶回開發機分析、commit、必要時更新 golden。

---

## 規則與邊界
- **Do**:import `qp/` 核心;從 layout 推導 bit-class;stub 測到綠才交;runner adapter 注入 + config 驅動 + 豐富 log;rotation 窮舉、per-vector 抽樣帶 CI;每步小提交。
- **Don't**:不重寫度量/注入;不用退役的 recompute 捷徑;不在開發機嘗試跑真 RaBitQ;不猜 bit-class 或 layout;不造假數字;不手改產物。
- 不確定就**回報並停**。

---

## 完成定義
- **你的交付(開發機)**:stub 測試全綠 + adapter-注入/config-驅動的 E1/E3a/E3b runner + runbook,可直接搬上工作站跑、無需在那邊改碼。
- **科學驗收(人類在工作站)**:跑出 RaBitQ vuln_map(含 rotation 是否災難性)、E3a/E3b 結果——這是 F1 與 Top-Down 削減順序的依據,由人類執行後帶回。

交付後回報「stub 全綠 + runbook 就緒」,等人類工作站跑完帶結果回來,再進分析與 Stage 2(scrub 機制 + temporal)。