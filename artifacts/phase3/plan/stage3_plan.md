# 指示:凍結前最後一輪(複本自我 scrub + 多 seed 分布 + F 驗證點 + G 分析)

> 四個任務:#1 複本自我 scrub(唯一的新機制,主圖第四條線)、#2 多 seed 保險絲首失敗分布(平行批次)、#3 F 合成的驗證點、#4 G 離線分析(零計算)。
> 工作流同前:import `qp/` 不重寫;report-and-stop;不造假;產物碼生成 + provenance 即時蓋(platform/host/seed/clean-baseline)。

---

## 任務 #1 · 複本自我 scrub(投票失敗即觸發)【最優先,主圖第四條線】

### 背景
run 4 證明:複本與主體同速累積損傷時,R=3 多數決是一次性保險絲——首次投票失敗(≥2 複本同 bit 損壞)後,因「絕不以錯誤多數覆寫資料」而拒絕回寫,複本永不再同步,其後 98/98 ticks 全部 irrecoverable,recall 沿無防護軌跡崩至 0。本任務實作修法,語意採**事件觸發**:投票失敗 = 行動信號(與斜坡層「CRC 失敗→fallback」、越界層「檢查失敗→跳過」同一哲學:偵測是行動的觸發器)。

### 實作(在 `RecoveryGuard._scrub_cliff` 的失敗路徑上)
1. **觸發式全量重載**:投票出現不可修復(無多數)→ 從 **clean 源**(索引檔內的靜態 rotation,64 B,serialized_region_map 可定位)重載 **主 buf + 全部 R 份複本**。
   - **必須全量**(64 B × 4),不是只修失敗的那個 bit:首次投票失敗時複本上已各自累積多個未同位的損傷(run 4 tick 2 為 [1,3,5] bits),只修失敗位置會留火種、旋即再失敗。全量 256 B 讀取近乎免費,直接做乾淨版本。
   - counters 新增:`cliff_reload_triggered`(觸發次數)、重載後驗證主 buf == clean。
2. **低頻保底(config 開關,預設開)**:每 N tick(預設【N=10,可調】)對主 buf 做一次對 clean 源的 64 B CRC 比對,不等即觸發同一個全量重載。
   - 目的:堵純觸發式的理論盲區——若 ≥2 複本**同位置同值**錯,多數決會投出錯值而無觸發信號(機率極低但論文會被問)。counters:`cliff_anchor_checked` / `cliff_anchor_mismatch`。
3. clean 源的讀取走 adapter(檔案 offset 讀 64 B),不要把 clean rotation 快取成第四份記憶體複本——**clean 源的意義是它在持久儲存、不受記憶體錯誤影響**;快取進 DRAM 就變成第四個會壞的複本,語意破功。code comment 標明這點。

### 驗證(stub,開發機)
- 構造 run 4 的失效情境(≥2 複本同 bit 損壞)→ 觸發 → 重載後主 buf 與複本全等 clean → 下一 tick 投票恢復正常(**不再**永久 irrecoverable)。
- 構造「同位置同值雙擊」(投票靜默出錯)→ 低頻保底在下個 anchor check 抓到並重載。
- 保底關閉時,上述情境**不**被抓到(驗證開關真的在控制行為)。
- 全量語意:觸發後複本上「其他位置」的累積損傷也被清掉(assert 複本 == clean,不只失敗位)。

### 工作站 run(runbook)
- 重跑 run 4 設定(`--inject-replicas` + 新 scrub 開啟),**200 ticks**(比 run 4 的 100 長,展示長期穩定;~1 h)。
- 預期:recall 全程 0.98376;`cliff_reload_triggered` > 0(保險絲情境確實發生過且被救回);irrecoverable 恆 0 或瞬時後歸零。
- **這條 timeline 就是主圖第四條線**(無防護崩 / 每查詢修全平 / 保險絲失效 / 自我 scrub 撐住)。
- 產出後與 run 2/3/4 合成主圖資料(四線對齊同 tick 軸)。

---

## 任務 #2 · 多 seed 保險絲首失敗分布【平行批次】

### 背景
run 4 的 tick-2 首失敗是單 seed(1234)抽樣,§5 統計注估計期望在 O(10) ticks。本任務把它變成分布。

### 實作
1. **先遷 SeedSequence**:複本 seed lane 由 `seed ^ tick ^ ((r+1)·0x5EED)` 改為 `numpy.random.SeedSequence(root).spawn(...)`(expb 曾在 XOR lane 抓到 aliasing;§8 限制 2 已建議)。遷移後 root seed=1234 的單 run 結果**允許與 run 4 不同**(lane 語意變了)——在報告中標明,不要為了重現舊值而保留壞 lane。
2. 批次 runner:**30 seeds**、`--inject-replicas`、**不帶 recall**(只記首次 irrecoverable 的 tick;分鐘級/run)、**平行執行**(每 seed 獨立進程,工作站可負擔;產物按 seed 分檔避免寫衝突)。
3. 另挑 2 個 seed(首失敗最早/最晚者)補帶 recall 的完整 timeline(2 × 27 min),確認「首失敗後沿無防護軌跡崩」的機制敘述在不同 seed 上成立。
4. 分析:首失敗 tick 的分布(中位數、IQR、min/max),對照 §5 的解析估計(以複本淨損 bits 估單 tick 失敗率)。

### 驗證
- 平行產物完整性:30 份 jsonl 各自 provenance 完整、無交叉寫入。
- 決定性:同 root seed 重跑 → 同分布。

---

## 任務 #3 · F 合成 + 驗證點

- 合成腳本(開發機):懸崖 timeline(run 2/3/4/#1)+ expb 容忍曲線與間隔比 + Stage 0 cost harness → frontier / 分析表(第四章解析表的數據源)。
- 挑 **3 個配置點**工作站實測確認(靜態:腐蝕至指定 f\* → 套 policy → 量 recall;每點分鐘級):
  (a) headline 點:R=0.9、EB、f=f\*_EB(0.101)→ recall 應 ≈ 0.90;
  (b) 對照:同 f 下 drop → recall 應 < 0.90(驗證間隔比的方向);
  (c) 懸崖點:rotation 累積 4 bits 無防護 → 應已 collapse(E3a 一致性)。
- 合成值 vs 實測值偏差 > 0.01 → report-and-stop,查內插。

## 任務 #4 · G 離線分析【零計算】

- 用 expb **現成** jsonl:對每 (pattern, f) 比對 `failed/checked`(on-access 觀察比例)vs 注入真值 f。
- 產出:一張小表/小圖(觀察 vs 真值,理想為對角線)+ 偏差統計。這是 §3.3「偵測直接吐出決策變數」的證據。
- 若現成欄位不足以重建 per-run 的 checked/failed → report-and-stop(不補跑,改為 limitation 一句)。

---

## 執行順序與凍結
1. #1 機制(stub 綠)→ 工作站 200-tick run【唯一可能出意外的新機制,先做先 smoke】。
2. #2 SeedSequence 遷移 + 30-seed 平行批次(可與 #1 的工作站時段並行)。
3. #3 合成(開發機,隨時並行)→ 3 驗證點(搭 #1/#2 的工作站時段)。
4. #4 純分析,插空做。
5. 全綠 → **實驗凍結**:此後發現的洞以文字/解析補,不回實驗。

## 規則與邊界
- **Do**:全量重載語意;clean 源走檔案不進 DRAM 快取;SeedSequence;平行產物分檔;provenance 即時蓋;每步小提交。
- **Don't**:不只修失敗 bit;不把 clean rotation 快取成第四複本;不為重現舊值保留壞 seed lane;不補跑 G;不開清單外的新實驗;不造假;不手改產物。
- 不確定就 report-and-stop。

## 完成定義
- #1:stub 驗證四項全綠;200-tick run recall 全平、觸發計數 >0、主圖第四條線資料到手。
- #2:30-seed 分布 + 2 條補充 timeline + 對照解析估計。
- #3:frontier 合成 + 3 驗證點偏差 ≤0.01。
- #4:觀察 vs 真值對照表。
- 四項齊 → 回報並宣告凍結,轉入寫作與圖表定稿。