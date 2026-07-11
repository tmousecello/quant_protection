# Stage 3 誠實性審計(第二輪,對抗性 + 多代理 code review)

日期:2026-07-10。方法:xhigh workflow code review(6 finders / 37 verifiers / 41 candidates,
9 項被證偽駁回)+ 審計者對凍結產物的離線重算。第一輪報告見 `STAGE3_AUDIT.md`。

**總結(相對第一輪的修正)**:第一輪「8/8 CLEAN」的機制層結論在嫌疑 1/2/3/6/7/8 上**經獨立路徑複核成立**,
但第一輪**漏掉一項真正的比較層 artifact**:主圖的 `fuse` 與 `self_scrub` 兩條線**不只差在 scrub 開關,
還差在複本注入 lane**(舊 XOR lane vs 新 SeedSequence lane)。這是配對比較的混淆變數,且方向偏向美化結論。

---

## 新發現(第一輪未捕捉)

### ARTIFACT-A:主圖 fuse vs self_scrub 的複本 lane 混淆

**證據(三條獨立線索互證)**

1. `phase3_e5_recovery.py:636` 現行碼以 `replica_lane_seed()`(SeedSequence)注入複本;
   遷移發生在 commit `5f7467a`(07-08 14:23),取代舊的 `seed ^ ((r+1)*0x5EED)` XOR lane。
2. 主圖 fuse 線的來源 `artifacts/phase3/e5/e5_uniform_accum_rotation_replicas.records.jsonl`
   **mtime 07-08 13:32**,早於遷移 commit 51 分鐘 → **run 4 是舊 XOR lane 產出**。
   self_scrub 線 `..._replicas_scrub.records.jsonl` mtime 07-09 19:27,新 lane。
3. commit `5f7467a` 的訊息自己寫明:「root 1234 的 --inject-replicas 結果與 run 4 合法不同
   (lane 語意已變)」;`phase3_e5_recovery.py:630-634` 的碼內註解重複同一句。

`phase3_f_synth.py:239-248` 把 `fuse -> RUN4`、`self_scrub -> SCRUB` 直接餵進主圖,
**只檢查檔案存在,不檢查 lane**。`run_stage3_x86.sh` 的 Phase A 只跑 `--inject-replicas --cliff-scrub`
(stem 帶 `_scrub`),**沒有任何 phase 會重新產生無 scrub 的 run 4**。

**影響範圍**

- 主圖「fuse 崩潰 vs self_scrub 全平」的**配對對比失效**:兩條線相差 2 個變數。
- **方向已知,且偏向美化**:遷移理由正是舊 XOR lane「aliasing-prone」(§8 限制 2,expb 曾抓到 lane 碰撞)。
  Aliasing 使多個複本更容易在**同一 bit 位置**同時髒 → 投票失敗更早 → fuse 崩得比無偏 lane 更早。
  亦即混淆的符號是**放大 fuse 的脆弱性**,而非削弱。
- 佐證(非證明):run 4(seed 1234,舊 lane)首次 irrecoverable 在 **tick 2**;
  30-seed 新 lane 批次的首失敗 **median 21、min 0**。tick 2 落在新 lane 分布的低尾,
  與「舊 lane 偏早失敗」一致。
- **不受影響者**:headline #2 的 30-seed 分布**全部使用新 lane**(`first_fail_summary.json`
  的 `seed_lane` 欄位即 SeedSequence),不受此混淆污染;self_scrub 自身 200 tick 全平是
  **run 內事實**,也不依賴 run 4。

**最小修復(凍結解除後)**:以新 lane、seed 1234、`--inject-replicas`(不加 `--cliff-scrub`)
重跑 100 tick 產生 run 4′,主圖改用 run 4′;或在圖說明文標註「fuse 線為舊 lane,
兩線差異包含 lane 變更,方向偏向放大 fuse 脆弱性」。前者為正解。

### ARTIFACT-B:Phase A 的 `cliff_irrecoverable == 0` gate 是套套邏輯

`phase3_e5_recovery.py:280-289`:`cliff_scrub` 開啟時,投票失敗走 `_cliff_vote_fail_reloads += 1;
_reload_cliff(buf); return` —— **在唯一的 `_cliff_irrecoverable += 1`(line 290)之前 return**。
故在 `--cliff-scrub` 下該計數器**結構上不可能遞增**,而 `run_stage3_x86.sh:78` 正是
`assert cliff_irrecoverable == 0` 作為健康證據。**這個 gate 永遠不會 fail**。

真正能反映「靜默服務錯誤值」的信號是 `cliff_anchor_mismatch`,它**只被列印,從未被 assert**。

**影響**:Phase A 的「全綠」對 headline #1 的支撐力低於表面。此 gate 檢查的東西與所聲稱的結論之間
**沒有推理距離**(審計原則 3 所指的套套邏輯)。**最小修復**:Phase A 改 assert
`cliff_anchor_mismatch == 0`,並保留 `cliff_vote_fail_reloads` 為資訊欄。

### ARTIFACT-C(次要):所有 sanity gate 皆為裸 `assert`

`run_stage3_x86.sh:58` 以 `"$PY" -c "$1"` 執行 gate body(未 `-E`、未清 `PYTHONOPTIMIZE`),
A/B/D 三個 gate 與 `phase3_e5_recovery.py:322` 的 post-reload 驗證、
`phase3_expb_recovery.py:307` 的 `crc_fail==n_elements` 釘死(G 對角線所倚)全是裸 assert。
若環境帶 `PYTHONOPTIMIZE=1`,**全部靜默消失且印出 OK**。凍結這一次未發生(logs 正常),
屬**潛在**而非已實現的 artifact。**最小修復**:gate 改 `if not cond: raise SystemExit(...)`。

---

## 原八項嫌疑:複核結論

| # | 嫌疑 | 結論 | 關鍵證據 |
|---|---|---|---|
| 1 | 注入沒碰到查詢路徑 | **CLEAN** | 見下 |
| 2 | scrub 在量測前偷跑 | **CLEAN**(順序即設計) | `phase3_e5_recovery.py:624→642`;scrub 在 `search_with_recovery` 內、serialize 之前 |
| 3 | clean 源被豁免 | **CLEAN**(合法威脅模型) | `adapter.py:82-97` 每次 `open()` 檔案讀,無快取;注入只打記憶體 buf/replica |
| 4 | 驗證點循環論證 | **gate a/b CLEAN;gate c ARTIFACT(措辭)** | 見下 |
| 5 | G 構造性傳播 | **機制 CLEAN;傳播已註記** | `g_detection_summary.json` 的 `note` 明載構造性 |
| 6 | seed 選擇效應 / 模型循環 | **CLEAN** | 見下 |
| 7 | 沒進報告的 run | **CLEAN** | 見下 |
| 8 | anchor 19/0 是否在比對 | **CLEAN(測試背書)+ 重新定性** | 見下 |

### 嫌疑 1 — CLEAN

主迴圈 `phase3_e5_recovery.py:622-642`:`corruptor.inject_step(buf, ...)`(624)注入**主 buf**,
`search_with_recovery(buf, tmp_path)`(642)內部先 scrub 再 `deserialize_index(buf, tmp_path)`(171)
再由 C++ 讀該 tmp 檔查詢。**被查詢的檔案由被注入的 buf 序列化而來**,不存在「兩份記憶體」。

最強反證來自產物本身:凍結 scrub run 的 `cliff_repaired = 2012` —— 若注入沒落到 buf/複本,
修復位元數會是 0。且 run 2 同一注入管線 recall 崩至 0.423。**注入確實命中,recall 全平是
「修復發生在序列化之前」的設計後果**,不是量到空氣。

### 嫌疑 4 — gate c 是重現性檢查,不是驗證點

`phase3_f_synth.py:393-394` 以 `{**FULL_CFG, "seed": config.SEED}` + `inject_step(buf, "uniform_accum", config.SEED ^ 0)`
重放 `phase3_e3c_temporal.py:261` 的 `seed = cfg["seed"] ^ tick` 在 tick 0 的**完全同一 lane**,
再與 run 2 自己的紀錄相減(401-402)。`cliff_check.json` 實測 `delta: 0.0`。
**決定性系統下 Δ=0 是必然**,`tolerance_curves` / `solve_fstar` 的任何合成 bug 都不會使它 fail,
但它被計入 `all_ok`(line 209)並印成「all 3 validation gates PASS」(line 497)。

只有 gate a/b 對合成有推理距離,且僅覆蓋 headline fraction。
**措辭修復**:「3 個驗證點」→「2 個內插驗證點 + 1 個決定性重現檢查」。

### 嫌疑 6 — CLEAN(模型非循環)

- seed 為 `range(seeds_start, seeds_start + n_seeds)`(`phase3_e5_seed_batch.py:289`),
  產物確認 **1000–1029 連號、n=30、無缺號**,全數進報告。
- `analytic_estimate(p=0.005, bits=512, R=3)`(line 88-105)以 `p_fail = C(R,2)·(bits·p)²/bits` 推出
  `p = 3 × 2.56² / 512 = 0.0384`。三個輸入全是**先驗物理量**:`p=0.005` 是注入器的每 bit 機率
  (`phase3_e3c_temporal.py:40` FULL_CFG),`bits=512` 是 64 B rotation 區,`R=3` 是複本數。
  **呼叫端 `analytic_estimate()`(line 188)不帶參數,沒有從 30 seeds 擬合任何東西**。非循環。
- 但「吻合」的措辭應收斂:解析 median **17.7** vs 實測 median **21**(差 19%);
  解析 mean 26.0 vs 實測 mean 25.4(而後者因 seed 1005 censoring 為**下偏**,不應拿來當吻合證據)。
  誠實說法是「同一數量級、量級一致」,不是「吻合」。

### 嫌疑 7 — CLEAN

`artifacts/phase3/logs/` 只有單一 timestamp 系列 `20260709_183839` 的 A/B/C/D 四個 log,
無中止痕跡、無其他 timestamp。`stage3_full.out` 108 行,無 `FAIL` / `Traceback` / `REPORT-AND-STOP`。
唯一「調參數」性質的 commit `2223d49`(gate 嚴格化)時間 07-09 18:24,**早於正式跑起始 18:38**,
方向是**把 gate 收緊**而非放寬 —— 與 p-hacking 相反。

### 嫌疑 8 — CLEAN,但 19/0 的證據價值需重新定性

`test_e3c_e5.py:472 test_anchor_catches_silent_wrong_vote` 確實**構造盲區情境**
(毒化記憶體內 CRC anchor 使投票驗證靜默通過),斷言 `_cliff_anchor_mismatch == 1`
且 reload 來自 anchor、並自癒被毒化的 anchor。現行碼上 **3 passed**。故 anchor **有在比對**。

**但**:凍結 run 的 19/0 幾乎不帶資訊。注入器只打 `buf` 位元組與 replica numpy 陣列,
**從不觸碰 Python int 的 `_rot_clean_crc` anchor**;要觸發盲區需 CRC-32 碰撞(~2⁻³²)。
換言之在此故障模型下 **anchor 本來就不可能 fire**,19/0 是**故障模型的構造後果**,
不是「機制在場上抓到了東西」的證據。論文不應把 19/0 當作 anchor 有效性的實測支持。

### 補充:嫌疑 1 第三支探針 — CLEAN(由第一輪離線重放釘死)

「tick 77/79/94/137 的投票失敗是否為真雙擊」**已被第一輪審計的離線 RNG 重放排除**
(`STAGE3_AUDIT.md` 嫌疑 1 證據 3):獨立複刻注入流後重放 200 ticks,預測的
「≥2 複本同位雙擊」tick 集合 = {77, 79, 94, 137},與 records 的 `cliff_vote_fail_reloads`
增量位置完全一致;預測 `cliff_repaired` 總數 2012 = records 終值。**本輪未重跑該重放**,
採信其結論;我獨立核對了 records 側的兩個數字(tick 集合、2012)確實如其所述。

順帶澄清一個**看似可疑但無害**的產物欄位:`replica_bits` 在 scrub run **200 tick 全為 0**。
原因是它記 `rc.cumulative_corruption()`(當前與 clean 相異的位元數)且**寫入時點在 scrub 之後**
(`phase3_e5_recovery.py:660-663`),而 `_scrub_cliff` 每 tick 把 majority 寫回所有複本(line 299)。
該欄位依構造恆為 0,**不承載 pre-scrub 複本狀態**——它不是證據,也不是 bug,只是無資訊量。
(對照:run 4 無 scrub,`replica_bits` 自 tick 2 首次投票失敗後開始累積,行為自洽。)

獨立的量級交叉檢查:每 tick 複本都被重同步 → 每 tick 是獨立 Bernoulli(p_fail) 試驗。
解析 `p_fail = 0.0384` → 200 tick 期望 **7.68** 次投票失敗,實測 **4** 次
(Poisson λ=7.68 下 P(X≤4) ≈ 0.10,約 1.3σ 偏低)。與重放結論相容。

### 對第一輪報告的一項更正

`STAGE3_AUDIT.md` 嫌疑 1 證據 2 寫道:「run 4 與 #1 是同一腳本、同一迴圈,**唯一分歧是
`cliff_scrub` 旗標**」。**這句話不成立**——run 4 另外還差一個複本注入 lane(見 ARTIFACT-A)。
第一輪的離線重放使用的是新 SeedSequence lane,故能重現 **scrub run**(新 lane)的 tick 集合;
它從未重現 run 4(舊 XOR lane),因此該混淆在第一輪未被觸及。

---

## 三個 headline 主張的證據強度分級

| 主張 | 分級 | 說明 |
|---|---|---|
| **#1 自我 scrub 可無限續命** | **直接量測(單 run)+ 配對對比受混淆** | 200 tick recall 恆 0.98376、`cliff_repaired=2012`、4 次投票失敗全數觸發重載復原 —— **run 內是直接量測**。但「相對 fuse 的優勢」所倚的主圖對比含 lane 混淆(ARTIFACT-A),且 Phase A 的 gate 為套套邏輯(ARTIFACT-B)。「無限」二字為**外推**,200 tick 未見劣化不等於無界。 |
| **#2 首失敗 median ~21** | **直接量測(30 seed,無選擇效應)** | 連號全收、新 lane、1 個 censoring 且遠大於 median。與解析幾何模型(先驗參數推導,非擬合)同數量級(17.7 vs 21)。**最紮實的一項**。措辭應為「量級一致」而非「吻合」。 |
| **#3 偵測零成本吐出決策變數** | **構造性** | `observed_load` 與真值逐點相等,分子由 `crc_fail == n_elements` 斷言釘死、分母由 `extract_points` 檢查。`g_detection_summary.json` 的 `note` 已誠實註記。**不是偵測準確率的實驗發現**,而是「偵測器在其設計假設下自洽」的驗證。論文任何引用 G 的地方(含圖說)都必須帶此限定。 |

---

## 誠實殘留:UNVERIFIABLE 清單

1. **`fstar_check` 與原 sweep 的時間差**僅由 mtime 推得,`meta` 無明文執行時間戳 / git commit 欄位。
2. **anchor 在真實故障模型下的效力**:stub 測試證明碼會比對,但無任何實測事件;19/0 不構成證據。
3. **舊 XOR lane 的 aliasing 強度未量化**:ARTIFACT-A 的混淆方向由「lane aliasing → 更早失敗」
   的機制論證與 run 4 tick-2 落在新 lane 分布低尾佐證,但**未實測**舊 lane 的 `p_fail`。
   要釘死方向與幅度需重跑 run 4′(凍結不解則維持推導級)。
