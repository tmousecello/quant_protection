# 指示:Stage 2 第一輪(E5 恢復機制 + E3c 累積模型)

> 建兩個 Stage 2 實驗的承重載體:**E5**(兩層恢復機制)與 **E3c**(temporal 累積故障模型)。兩者**可平行建構與單元測試**;它們在下游的實驗 A/B 才合在一起。**先把交會的介面契約(§1)定死**,兩邊才能平行做、最後無縫接上。沿用既有工作流:開發機 + stub 開發,真跑在 x86 工作站,import 不重寫,report-and-stop,不造假。

---

## 1. 介面契約(先定,兩邊平行的前提)
E3c 與 E5 在實驗 A/B 的查詢迴圈裡交會。固定這個接縫:
```
for tick in timeline:
    e3c.inject_step(index, region, pattern)        # 累積一步腐蝕(不還原)
    for q in queries:
        r = e5.search_with_recovery(index, q)       # 偵測+恢復後查詢
    m = measure(results)                            # recall@10、is_silent_collapse(import qp)
    record(tick, e3c.cumulative_corruption(), m, e5.counters())
    e5.scrub_if_due(tick)                           # rotation 每查詢修 / ex 到閾值批次 reload
```
契約:
- **E3c 提供**:`inject_step(index, region, pattern, seed)`、`cumulative_corruption()`(回傳目前累積腐蝕 bit 數 / 比例)。
- **E5 提供**:`search_with_recovery(index, query)`、`counters()`(checked/failed、known-corrupted)、`scrub_if_due(tick)`。
- 兩者都吃**同一個 adapter + region map**(Stage 0 的)。契約固定後,E3c 與 E5 各自獨立開發。

---

## 2. 工作流(同前)
- **你只在開發機(arm64)工作,對 stub adapter 開發與單元測試。RaBitQ 真跑在 x86 工作站,由人類執行。**
- runner / 機制都 **adapter 注入 + config 驅動**,工作站上不改碼。
- 豐富 log + 中間狀態,讓工作站(無 Claude)失敗時可帶回開發機除錯。
- **import 不重寫**:`qp.metrics`(`is_silent_collapse`,排除 crash 與 nan-inf)、`qp.faults`(`uniform_p`/`spatial_cluster`/`cross_row`/`temporal_burst`/`CumulativeCorruption`)、**以及 Samuel 既有的 EB-aware fallback**(見 §5,slope 層要 import 它、不要重寫恢復政策)。
- report-and-stop;不造假;產物碼生成。

---

## 3. 開始前:讀這些
- Stage 0/1 成果:`qp/rabitq/layout.py`+`adapter.py`(region map、serialize/deserialize)、`qp/faults.py`、`qp/metrics.py`、Stage 1 的 criticality / vuln_map。
- **Samuel repo 的 EB-fallback 恢復碼**(exp 3 的 hero 機制:用 bin + error bound 悲觀排序)——slope 層要 import/包裝它。
- 回報你的理解 + E5/E3c 的平行實作 plan。

---

## 4. 背景(精簡)
組織原則 = **懸崖 vs 斜坡**,兩者做相反的事:
- **rotation(懸崖,64 B)**:慢性累積、4–8 bit 沉默崩潰硬閾值、0% crash。全域、每查詢必用。
- **ex(斜坡,96 MB)**:可偵測緩降、有 bin 當免費 fallback。per-vector、只在 rerank 時碰到。
E5 對這兩類做相反的恢復;E3c 提供讓它們隨時間累積壞掉的故障流。下游 A 量懸崖的 time-to-cliff、B 量斜坡的 recall 緩降與 EB 買時間。

---

## 5. 要建的東西

### E3c · temporal 累積故障模型
**build**(基於 `qp.faults` 的 `CumulativeCorruption`,faiss-free):
- timeline 驅動:每 tick 注入腐蝕並**累積**(不還原),可指定 region。
- 兩種累積樣式:
  - `uniform_accum`:每 tick 依 `uniform_p` 注入(各 bit 獨立 p)。
  - `clustered_accum`:每 tick 依 `spatial_cluster(W,k)` / `cross_row` 注入(rowhammer 式叢集)。
- 可選疊加:靜默期 + 觸發點 burst(`temporal_burst`)。
- 狀態:追蹤累積腐蝕(以**物理 XOR 狀態**為準——同一 bit 翻兩次=回到 clean;`cumulative_corruption()` 回傳目前與 clean 相異的 bit 數/比例)。決定性(同 seed 同軌跡)、可 reset。

**驗證**:
- 累積單調:相異 bit 數隨 tick 非遞減(忽略罕見的雙翻抵銷),且速率符合(uniform ≈ N·p/tick;clustered = k/window/tick)。
- 樣式差異:在 64 B rotation 上,`clustered_accum` 與 `uniform_accum` 的**空間分布確實不同**(叢集集中、uniform 散布)——assert 分布,不是 assert「誰先到崖」(那是實驗 A 的結果)。
- 雙翻語意:同 bit 翻兩次後 `cumulative_corruption` 不把它算成腐蝕(物理 XOR 正確)。
- 決定性 + reset 回 clean。

### E5 · 兩層恢復機制

#### 5a · 懸崖層(rotation:複製 + 多數決)
**build**:
- 維護 R 份(預設 3×)rotation 複本(64 B)。
- `scrub_if_due`:比對複本、多數決偵測+修復不一致、回報不一致 bit 數(= 精確錯誤計數)。預設**每查詢**做(rotation 全域、每查詢必用 → on-access=完整覆蓋)。
- 恢復:多數決重建正確 rotation、修復被腐蝕的複本。
- **絕不用「parity + 超出才補救」**。

**驗證**:
- 對 1 份複本注入 k bit(k < 多數)→ 多數決重建正確值、回報不一致數 = k、修復後複本全等 clean。
- 邊界:若 ≥ 多數的複本在同位置被腐蝕(無法多數決)→ **偵測到但回報「不可修復」,不靜默服務錯值**。assert 這個邊界被標記、不是默默放過。
- 每查詢修 → 累積永遠到不了 4-bit(用 E3c 餵 rotation,驗證 recall 持平不墜崖)。

#### 5b · 斜坡層(ex:CRC chunk + EB-fallback + lazy batch reload,parity 開關)
**build**:
- 把 ex_data 切 chunk(大小 config,預設對齊 IO 粒度如 4 KB),每 chunk 一個 CRC。
- rerank 走到某向量時驗其 ex chunk CRC:
  - pass → 用精確 ex 距離。
  - fail → **EB-fallback(import Samuel 的恢復政策,不重寫)**:用受保護的 bin + error bound 悲觀排序;記入 known-corrupted set;`failed`++。
- `checked`/`failed` 計數;`failed/checked` = 當前腐蝕比例估計。
- lazy batch reload:腐蝕比例逼近容忍線(config,如 R=0.9 的 ~10%)→ 批次「reload」被腐蝕 chunk(模擬 = 從 clean 副本還原)→ 重置 counter。
- **parity 開關(預設 off)**:on 時先試從 parity 修 chunk;超出 parity 容錯才 EB-fallback + 標記 reload。

**驗證**:
- 腐蝕一個 ex chunk → CRC 抓到 → EB-fallback 觸發,且**排序與 Samuel 既有 EB 路徑一致**(import 正確) → 進 known-corrupted、`failed`++。
- 腐蝕比例過閾值 → batch reload → counter 重置、recall 回升。
- `failed/checked` 正確追蹤注入的腐蝕比例(這是 §G 驗證的 counter 基礎)。
- parity on:容錯內的腐蝕被 parity 修(不觸發 fallback);超出 → fallback + 標記 reload。

#### 5c · bounds-check skip(pointer / links / cluster_id)
**build**:遍歷時對 pointer/index 做範圍檢查;越界 → **skip(不 crash、不丟整個結構)**。
**驗證**:把 pointer 腐蝕成越界 → 被攔截 → skip,無 crash → recall 維持(對比沒 bounds-check 會 crash)。

---

## 6. Stub 測試(開發機驗收 = 你的交付門檻)
對 stub + 已知注入 assert:
- E3c:累積單調、樣式差異、雙翻語意、決定性。
- E5 懸崖:多數決修復 + 精確計數 + 不可修復邊界被標記。
- E5 斜坡:CRC 抓到 → EB-fallback(與 import 的政策一致)→ counter → 過閾值 reload 重置;parity 開關行為。
- E5 bounds-check:越界 skip 不 crash。
- 介面契約(§1)可被 A/B 的迴圈呼叫(用 stub 跑一個 mini timeline 驗證 E3c+E5 能組起來)。
- config 切 stub↔真 adapter 不需改碼;決定性。

---

## 7. 給人類的 runbook(x86 工作站)
1. 同步上工作站,config 用真 adapter。
2. **冒煙**:對 rotation 跑一個短 timeline,確認 E5 懸崖層每查詢修、recall 持平;對 ex 跑一個短 timeline,確認 CRC+EB+reload 迴圈動作、counter 合理。
3. 這兩個冒煙過,Stage 2 第一輪載體就緒,可進實驗 A(rotation time-to-cliff)、B(ex 斜坡 + EB 買時間)。
4. 帶 log / 中間產物回開發機。

---

## 8. 規則與邊界
- **Do**:import `qp` 與 **Samuel 的 EB-fallback**;懸崖每查詢修、斜坡 on-access piggyback;先定 §1 契約再平行;每步小提交。
- **Don't**:不重寫度量/注入/EB 政策;不在開發機跑真 RaBitQ;rotation 不用 parity-被動模型;不造假;不手改產物;不跑實驗 A/B 的完整 sweep(那是下一輪、在工作站)。
- 不確定就 report-and-stop。

---

## 9. 完成定義
- **你的交付(開發機)**:E3c 與 E5(三層)建好、stub 測試全綠、§1 契約可組合、config 驅動、runbook 就緒。**平行建構,最後用 mini-timeline stub 測驗證兩者接得起來。**
- **科學驗收(人類,x86)**:冒煙過 → 進實驗 A/B。

交付後回報「E5+E3c stub 全綠 + 契約可組合 + runbook 就緒」,等工作站冒煙與實驗 A/B。