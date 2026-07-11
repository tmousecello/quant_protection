# Stage 3 結果誠實性審計(對抗性)

**日期**:2026-07-10 **審計者角色**:對抗性——目標是證明 Stage 3 結果為量測 artifact,審計失敗(所有攻擊被證據擋下)結果才算通過。
**方法邊界**:只讀凍結產物與碼;唯二的主動驗證是 (1) 純 numpy 離線 RNG 重放(不 import 工作碼、不碰凍結產物、不重跑實驗,腳本見附錄)、(2) stub 測試套件執行(秒級、stub adapter)。
**總結**:**8 個嫌疑全數 CLEAN(機制層零 artifact),但抓到 3 處措辭層問題**——「太乾淨」的三個數字各有一個誠實的結構性解釋,其中兩個(recall 全平、G 逐點相等)本來就是設計後果/構造性,不應在摘要層被當成獨立實驗證據陳述。逐項如下。

---

## 嫌疑 1【最高】:#1 的 recall 恰等 clean——注入是否碰到查詢路徑?

**結論:CLEAN(注入確實打到查詢路徑;離線重放逐 tick 命中)+ 一項措辭層再定性。**

證據:

1. **注入與查詢共用同一份資料,無逃逸副本**。注入 XOR in-place 打記憶體 `buf`(`phase3_e3c_temporal.py:185`);每 tick `deserialize_index(buf, tmp_path)` 把 `buf` 寫成 tmp 檔(`qp/rabitq/adapter.py:71-73`),C++ `exp_dumpids` subprocess 從該 tmp 檔新載搜尋(`adapter.py:194-196`)。序列化發生在 scrub 之後(`phase3_e5_recovery.py:168→171`),順序正確(inject → repair → serialize → search),不存在「查詢端持有未注入副本」。
2. **run 4 對照成立**:run 4 與 #1 是同一腳本、同一迴圈(`phase3_e5_recovery.py:622-666`),唯一分歧是 `_scrub_cliff` 投票失敗分支的 `cliff_scrub` 旗標(`:279-293`)——scrub off 時 t2 首失敗後 recall 沿無防護軌跡崩落。同管線在旗標關閉時確實崩 → 管線能影響查詢。
3. **離線 RNG 重放(本審計最強證據)**:注入完全決定性(main lane `root^tick` `:623`、replica lane `SeedSequence([root,tick,r+1])` `:62-77`、`uniform_p` 的 `default_rng→binomial→choice` 序列 `qp/faults.py:70-77`)。獨立複刻後重放 200 ticks:
   - 200 個 tick 的 main `bits_flipped` 與 records **逐 tick 全等**(重放忠實性自證);
   - 預測「≥2 複本同位雙擊」tick = **{77, 79, 94, 137}**,與 records 的 `cliff_vote_fail_reloads` 增量位置**完全一致**——4 次投票失敗是注入流的真實雙擊,非隨機噪音,且證明注入確實進入了投票的複本結構;
   - 預測 `cliff_repaired` 總數 = **2012** = records 終值(每一個注入 bit 都有帳,無漏無虛)。

**再定性(進論文措辭)**:「200 ticks recall 全平」本身**不是**機制成功的獨立證據——它是設計後果。scrub 語意下 rotation 在每次查詢前必被修復為 clean bytes(多數決寫回 `:297` 或全量重載 `:315`),故 recall 恆等 clean 是決定性相等;run 1(無複本無 scrub,多數決永遠成功)同樣全平 0.98376,recall 無法區分「scrub 救了 4 次」與「scrub 從未觸發」。承重證據是 counters(`vote_fail_reloads=4`、`irrecoverable=0`)+ run 4 對照。§3 表格把 recall 全平列為第一指標,建議加註「設計後果:修復於量測前完成;證明機制工作的是 reload/irrecoverable 計數」。

## 嫌疑 2:scrub 是否在量測前偷跑?

**結論:CLEAN(順序即機制設計,與敘述一致)。**

迴圈實際順序(`phase3_e5_recovery.py:622-666`):inject main(:624)→ cc 量測 pre-repair(:625)→ inject replicas(:627-636)→ `search_with_recovery`(:642,內部 `_pre_search_scrub` → serialize → search)→ recall(:644)→ anchor(:653)→ slope scrub(:666)。「注入 → 修復 → 量測」正是「每查詢修」的論文語意(`_scrub_cliff` 每查詢無條件跑,非量測前一次性),無敘述與實作不符。run 4 同迴圈只差旗標,run 4 的 t0/t1 recall 也等 clean(投票成功期間),t2 失敗後才崩——行為與順序自洽。records 的 `cumulative_corruption` 量在修復前(:625),per-tick 損傷確實每 tick 發生(1–6 bits),不是注入沒生效。

## 嫌疑 3:clean 源豁免——機制假設還是實驗捷徑?

**結論:CLEAN(含 sha256 完好性驗證)。**

1. 重載來源:`_reload_cliff`(`phase3_e5_recovery.py:314`)→ `read_serialized_range`(`qp/rabitq/adapter.py:82-97`):每次呼叫重開 `INDEX_PATH`、seek 到 rotation offset 讀 64 B,註解明文「Deliberately a fresh file read on every call — NEVER cached」。無 DRAM 快取:`_clean_buf` 只用於 OOB restore(:507),且 `test_reload_reads_persistent_source_not_dram_snapshot`(`tests/test_e3c_e5.py:425`)毒化 `_clean_buf` 後斷言重載仍還原持久源。構造子在 adapter 無 `read_serialized_range` 時直接 REPORT-AND-STOP 拒絕退回記憶體快照(:107-111)。
2. 注入目標與 clean 源分離:注入只碰記憶體(`buf` + 複本陣列);查詢寫的是 `tempfile` tmp 檔(:606-607),**任何路徑都不寫 `INDEX_PATH`** → 「檔案不受 DRAM 錯誤影響」的威脅模型假設與實作一致。
3. **完好性離線驗證(本審計新增)**:`INDEX_PATH`(`hnsw_M16_efC200_b7.index`)現行 sha256 = `fed2b245…dd26`,與 07-08 expb provenance 記錄的 `index_sha256` **完全一致**(`expb_meta.json:76`)——clean 源不是「定義上乾淨」而是可驗證未變動。

## 嫌疑 4:#3 驗證點的循環論證?

**結論:gate (a)(b) CLEAN(真實新量測,非循環,但資訊量有限);gate (c) ARTIFACT-輕度(措辭層)——它是重現性檢查,不是驗證點。**

1. **fstar_check 是新 run,非重讀**:新 fraction 0.101(原 sweep 只有 0.01/0.05/0.08/0.2/0.57)、新損壞索引 sha(`d16d35c3…`,原 sweep 5 個各異)、404,000 個新注入 flips、mtime 晚原 sweep 約 31 小時(07-09 20:12 vs 07-08 13:32)。**但 seed 同 1234、同 clean 索引** → 是「同一決定性系統在新操作點的新量測」,不是換 seed 的獨立複本。
2. **(a)(b) 驗什麼**:合成預測 = light 曲線在 0.08→0.2 錨點之間對 f=0.101 的線性內插(`phase3_f_synth.py:271-274`);實測 = 新 sweep 在 0.101 的 4-pattern 平均。偏差 0.000186/0.001072 小,合理解釋是平滑單調曲線在量測錨點附近內插精度高——**非套套邏輯**(實測完全可能落在門檻外),但它驗證的是內插正確性,不是容忍曲線的獨立佐證。
3. **(c) Δ=0.0 是數學必然**:`run_cliff_check`(`phase3_f_synth.py:390-406`)刻意用同 corruptor、同 cfg、同 seed lane(1234^0)重注入**同 6 bits**;決定性系統下 recall 必等 run-2 tick-0 的 0.42309。碼註解與 `cliff_check.json` note 都自認(user-approved deviation),§5.1 內文也如實寫「決定性 6-bit 重現」——**但 TL;DR(`STAGE3_ANALYSIS.md:26`)把它計入「3 個工作站驗證點全過」**,措辭過強。
   - **影響範圍**:無數據失效;僅「驗證點」計數的修辭強度。
   - **最小修復**:TL;DR 與 §5.1 標題改為「2 個內插驗證點(偏差 ≤0.0011,門檻 1/9)+ 1 個決定性重現檢查(Δ=0.0,按構造必然)」。

## 嫌疑 5:G 的構造性定位是否誠實傳播?

**結論:機制 CLEAN;傳播 ARTIFACT-輕度(TL;DR 與完成定義表漏限定)。**

1. 斷言本體在 `phase3_expb_recovery.py:307`(`assert got == n_elements`):裸 assert、不在任何 try/except 內、runner 用 `.venv/bin/python` 無 `-O`(`run_stage3_x86.sh:28`)→ **失敗會中止 sweep(誠實 abort),不存在被吞掉的選擇效應**。分母端 `phase3_g_detection.py:62-70` 不符即 `raise RuntimeError`。
2. 構造性限定的傳播盤點:`g_detection_summary.json` note ✅、`detection_table.md` 開頭 ✅、圖子標題(「decision variable / informational」`phase3_g_detection.py:195-196`)✅、`STAGE3_ANALYSIS.md` §6(:170-175)✅;**TL;DR(:29-30)❌、完成定義表(:186)❌**——只寫「完全相等/逐點相等、§3.3 證據成立」,只看摘要會誤讀為經驗性偵測準確率發現。
   - **影響範圍**:無數據失效;讀者對 G 證據性質的誤判風險。
   - **最小修復**:TL;DR #4 加「(構造性:分子由 run 內斷言釘死;價值在『偵測層免費產出決策變數 f』的論證)」;完成定義表同步。

## 嫌疑 6:#2 的選擇效應與模型吻合度

**結論:CLEAN(且經重放全數重現)。**

1. **無選擇效應**:seed 由 `range(1000, 1030)` 連號生成(`phase3_e5_seed_batch.py:289`),無跳過/重試;單 seed 失敗會 raise 中止整批(:152-165),不存在「跑壞悄悄換 seed」機制;`first_fail_summary.json` 30 個全數入帳。變體檔已定位:`seed1000r2` = 決定性 gate(與原版 byte-identical,`cmp` 驗證)、`seed1002full/1003full` = earliest/latest 補充 timeline(:246-248),`STEM_RE` regex(:33)結構上排除於統計,三者報告皆點名(:39、:184、:220)。
2. **p=0.0384 非循環**:`analytic_estimate`(:88-105)= C(3,2)·(512×0.005)²/512,輸入全是注入設定常數(p=0.005 即實際注入率,`phase3_e3c_temporal.py:40,150`;512 bits = 64 B rotation;R=3),**與 30 seeds 的失敗數據無共用輸入**。測試(`test_stage3.py`)把 geometric_mean 釘在獨立常數 26.042(非 `1/p_fail` 恆等式)並在 R=4 下判別 C(R,2),防退化。
3. **重放重現(本審計新增)**:用注入物理模型離線重算 30 個 root 的首個「≥2 複本同位碰撞」tick,**30/30 與 `first_fail_summary.json` 完全一致,含 seed 1005 在 150 ticks 內無碰撞(censored)**——整個 #2 分布可從注入參數獨立重導,「實測 vs 模型同量級吻合」是真驗證。censored 處理如實(median/IQR 只算 observed、mean 標下偏,`STAGE3_ANALYSIS.md:195-196`)。

## 嫌疑 7:有沒有沒進報告的 run?

**結論:CLEAN。**

1. `artifacts/phase3/logs/` 僅一組 timestamp `20260709_183839`(A→D 連續 19:27–20:15);`stage3_full.out` 單一 header、無 Traceback/REPORT-AND-STOP/重跑拼接,末行正常收尾。無正式跑中止痕跡(`e3c_full_run.log`/`e5_full_run.log` 為 07-03 Stage 2 時代)。
2. git 07-08→07-09 唯一碰 gate 的 commit `2223d49`(正式跑前 14 分鐘):方向是**收緊**——f_synth crash row 由靜默略過改 REPORT-AND-STOP、`c_cliff` 增加 ref 非 None 檢查(修 vacuous pass)、測試恆等式斷言換獨立釘值、Phase A gate 新增 `vote_fail_reloads>0`。無放寬 tol(始終 0.01)、無改注入參數。是修 bug/提高證據強度,非 p-hacking。
3. 未被報告引用的產物盤點:expb 中間 scratch 檔(`_expb_*.index` 等,非結果)、`artifacts_smoke/`(smoke,seed 1000/1001 的 full 變體屬 smoke 資料集,與正式 1002/1003 不同為正常)。`figures/fig_*.png` 三張(mtime 07-10 03:11)由凍結後的 `phase3_stage3_figures.py` 只讀產物生成,即 STAGE3_ANALYSIS.md 自身嵌的三張圖(:69、:99、:141),與凍結宣告(圖表定稿期)一致。**無失敗 run 被藏起來的跡象。**

## 嫌疑 8:anchor 保底 19/0——真的在工作嗎?

**結論:CLEAN(stub 測試存在、構造盲區、現行碼通過)+ 一項誠實加註建議。**

1. `test_anchor_catches_silent_wrong_vote`(`tests/test_e3c_e5.py:472-502`)確實構造盲區:2 of 3 複本同位翻轉(`:477-478`)+ 毒化 in-memory CRC 使投票靜默通過(`:479-482`),先斷言錯誤多數已寫入 buf 且無 reload(`:485-487`),再斷言 anchor 從持久源抓到 mismatch、觸發重載、buf 復原、毒化 CRC 自癒(`:495-502`)。**現行碼上 38/38 全過**(含 `test_anchor_no_false_trigger_on_repaired_buf`、`test_anchor_switch_off_leaves_silent_error`);`test_stage3.py` 36/36 亦過。
2. **加註建議**:run 內 19/0 本身資訊量趨近零——本 run 的注入範圍(rotation bytes)構造上打不到 in-memory CRC(Python int,非注入目標),CRC32 碰撞機率 ~2⁻³²,故 19/0 是預期值,無法與「恆回傳 OK」區分。anchor 有效性的證據是 stub 測試(同一 code path),不是 19/0。§3 表格「低頻保底運轉正常」建議補「(0 mismatch 為本 run 注入範圍下的預期;有效性由 stub 測試背書)」。

---

## 三個 headline 主張的證據強度分級(進論文措辭校準)

| 主張 | 分級 | 說明 |
|---|---|---|
| **自我 scrub 可無限續命** | 機制事件:**直接量測**(且經離線重放獨立驗證);recall 全平:**設計後果**(不可當獨立證據);「無限」:**機制推論** | 4 次投票失敗/重載、irrecoverable=0 是直接量測,雙擊位置經 RNG 重放逐 tick 命中;「≥200 ticks、>200× 下界」是量測,「可無限續命」是「每次失敗都可從持久 clean 源再生」的機制論證(外推),措辭應維持「機制上可無限續命」而非「實測無限」 |
| **首失敗 median ~21 ticks** | **直接量測 + 獨立推導雙重支撐**(三者中最強) | 30 連號 seed 全數入帳、無選擇效應;幾何模型 p=0.0384 純由注入參數推導(非擬合);審計重放 30/30 重現全部首失敗 tick(含 censored) |
| **偵測零成本吐出決策變數** | **構造性展示**(產物與 §6 已自認;TL;DR 需補限定) | 逐點相等由 run 內斷言 + 分母守衛構造保證;主張的真正內容是「CRC 偵測層在正常服務路徑免費產出 f,可直接對 f\* 表排程」——這是論證/展示,非經驗性準確率發現;access 加權版的 ~10⁻⁴ 偏差才是實測觀察 |

## 措辭層修復清單(全部凍結相容,只改文字)

1. `STAGE3_ANALYSIS.md` TL;DR #4 與完成定義表 :186:補「(構造性)」限定(嫌疑 5)。
2. TL;DR #3:「3 個驗證點」改「2 個內插驗證點 + 1 個決定性重現檢查」(嫌疑 4)。
3. §3 表格 recall 全平行:加「設計後果」註(嫌疑 1 再定性);anchor 19/0 行:加「預期值/stub 測試背書」註(嫌疑 8)。

## UNVERIFIABLE 殘項(誠實列出,均非結論性風險)

- fstar_check 與原 sweep 的 31 小時時間差僅由檔案 mtime 推得(meta 無明文 timestamp/git commit 欄位);若要釘死,provenance 應加執行時間戳(文字層建議,不影響「新 fraction + 新損壞索引 = 新量測」的結論)。
- 幾何模型的「每 tick resync 後新鮮 Binomial」假設未由 records 逐 tick 直接量測(`replica_bits` 是 post-scrub 殘餘)——但審計重放已用同一假設逐 seed 命中全部 30 個首失敗 tick,間接閉合。

## 附錄:離線 RNG 重放腳本(獨立複刻,未 import 工作碼)

```python
import json
import numpy as np

N_BITS = 512   # rotation region: 64 B
P = 0.005      # cfg.get("p", 0.005) default — e5 cfg carries no "p" key
ART = "artifacts/phase3"

def sample(seed):                                  # == qp.faults.uniform_p RNG sequence
    rng = np.random.default_rng(seed)
    count = int(rng.binomial(N_BITS, P))
    return set(int(b) for b in rng.choice(N_BITS, size=count, replace=False))

def replica_lane_seed(root, tick, r):              # == phase3_e5_recovery.py:62-77
    return int(np.random.SeedSequence([int(root), int(tick), int(r) + 1])
               .generate_state(1, dtype=np.uint64)[0])

def tick_replay(root, tick):
    main = sample(root ^ tick)                     # main lane, phase3_e5_recovery.py:623
    reps = [sample(replica_lane_seed(root, tick, r)) for r in range(3)]
    hit, collision = {}, False
    for r in range(3):
        for b in reps[r]:
            hit[b] = hit.get(b, 0) + 1
            if hit[b] >= 2:
                collision = True                   # >=2 copies dirty at same bit -> vote fail
    return main, reps, collision

# A. #1 scrub run: per-tick bits, vote-fail ticks, repaired total
recs = [json.loads(l) for l in open(f"{ART}/e5/e5_uniform_accum_rotation_replicas_scrub.records.jsonl")]
pred_fail, pred_repaired, mismatches = [], 0, 0
for t in range(200):
    main, reps, collision = tick_replay(1234, t)
    mismatches += (len(main) != recs[t]["cumulative_corruption"]["bits_flipped"])
    if collision: pred_fail.append(t)
    else: pred_repaired += len(main) + sum(len(r) for r in reps)
# 結果:mismatches=0;pred_fail=[77,79,94,137](==records 增量);pred_repaired=2012(==records 終值)

# B. 30-seed fuse first-fail
summary = json.load(open(f"{ART}/e5_seeds/first_fail_summary.json"))
for root in range(1000, 1030):
    pred = next((t for t in range(150) if tick_replay(root, t)[2]), None)
    assert pred == summary["first_fail_tick_per_seed"][str(root)]
# 結果:30/30 全等(含 1005 censored=None)
```

執行環境:`.venv/bin/python`(numpy 2.4.6,與正式跑同版)。stub 測試:`.venv/bin/python -m pytest artifacts/phase3/tests/test_e3c_e5.py`(38 passed)、`test_stage3.py`(36 passed),2026-07-10 本機執行。
