# E3c / E5 結果詳細分析 — 時序累積腐蝕 × 兩層恢復機制(rotation 懸崖)

**日期**:2026-07-04(x86-64 工作站實測)
**索引**:RaBitQ-HNSW b=7、SIFT1M、M=16、efC=200、L2;查詢 k=10、ef=2000(plateau)
**乾淨基線**:recall@10 = **0.98376**(Stage 0 錨點,本輪四組 run 均重現)
**共同條件**:pattern = `uniform_accum`(每 tick 每 bit 以 p=0.005 獨立翻轉)、region = `rotation`
(FhtKacRotator 全域符號向量,64 B = 512 bits)、seed = 1234、100 ticks。
recall 一律由 `qp.metrics.recall_at_k` 對 C++ 傾印的 ids 計算(ids-only 硬規則)。

本文整合四組真實(非 stub)runs:

| # | Run | 產物 | 說明 |
|---|---|---|---|
| 1 | E3c 注入(僅注入) | `e3c/e3c_uniform_accum_rotation.records.jsonl` | 腐蝕動力學本身 |
| 2 | E3c `--measure-recall` | `e3c/e3c_uniform_accum_rotation_recall.records.jsonl` | **無恢復** per-tick recall 時間線(新) |
| 3 | E5 兩層防護 | `e5/e5_uniform_accum_rotation.records.jsonl` | 防護下的 per-tick recall + counters |
| 4 | E5 `--inject-replicas` | `e5/e5_uniform_accum_rotation_replicas.records.jsonl` | R=3 複本同時累積(新;堵「複本不朽」caveat) |

Run 2、4 是本輪新增旗標的首次實測;1、3 與 7/3 的原始 run 完全一致
(run 1 依決定性重生,final 165 bits 與原 summary 吻合)。

---

## 1. TL;DR

1. **無恢復時,rotation 累積腐蝕在第 1 個 tick 就靜默崩潰**:6 bits(佔 rotation 0.117%、
   佔整個索引 2.7×10⁻⁶%)→ recall 0.98376 → 0.423,已越過 collapse 線(< 0.5×clean)。
   原先「≤2 ticks 到懸崖」的推導,實測收緊為 **tick 0 即崩**。全程 100 ticks 無 crash、
   無 nan-inf —— 純 silent-wrong,系統毫無外顯徵兆。
2. **E5 兩層防護把壽命從「不到 1 tick」延到「≥100 ticks 全程無損」**:recall 100 個 tick
   全部恰為 0.98376(單一值),235/235 個注入 bit 全數修復,壽命增益 **>100×**(下界,
   實驗到 100 ticks 為止)。防護成本極低(64 B × R=3 副本 + 每 tick 512-bit 投票)。
3. **但多數決的保護在「複本同速累積」下是一次性的**:tick 2 首次多數決失敗
   (≥2 副本同 bit 損壞)後,因「拒絕服務錯值 → 不回寫 → 副本永不再同步」,防護
   **永久失效**(其後 98/98 ticks 均 irrecoverable),recall 沿無防護軌跡崩至 0.0。
   多數決需要搭配「副本自身的定期驗證重載」(或 ECC)才不是一次性保險絲。
4. E3c 的 XOR 累積動力學符合理論:235 次原始翻轉在 512-bit 區域淨留 165 bits
   (奇偶模型期望 154,同量級),4 個 tick 出現淨值下降(重複命中互消)。

---

## 2. E3c:腐蝕動力學(run 1)

### 2.1 注入過程統計

- 每 tick 翻轉數:平均 **2.35**(min 0、max 8),與 Binomial(512, 0.005) 期望 2.56 一致
  (100 ticks 抽樣噪音內)。
- 100 ticks 累計**原始**翻轉 **235** 次;**淨**損壞(與 clean XOR 後 popcount)**165 bits**
  = rotation 的 **32.2%**。
- 差額 70 bits 來自重複命中同一 bit 的互消(XOR-toggle 語意):奇偶模型
  E[淨] = 512·(1−(1−2/512)^235)/2 ≈ **154**,觀測 165,同量級吻合;曲線有 **4 個 tick 淨值下降**
  (double-flip cancel 的直接證據)。物理意義:同一 cell 再次翻轉會「修回」,因此高密度下
  淨腐蝕成長是次線性的 —— 對「以 faults/MB 正規化」的外推要用淨值而非原始注入數。

### 2.2 淨腐蝕里程碑

| tick | 0 | 2 | 9 | 24 | 49 | 74 | 99 |
|---|---|---|---|---|---|---|---|
| 淨 bits | 6 | 13 | 31 | 56 | 102 | 125 | 165 |
| 佔 rotation | 1.2% | 2.5% | 6.1% | 10.9% | 19.9% | 24.4% | 32.2% |

`reset()` 後 drift = 0(注入簿記可逆性驗證通過)。

---

## 3. 無恢復基線:per-tick recall 時間線(run 2,新量測)

這是實驗 A 收尾清單裡「把 ≤2 ticks 從推導變直接量測」那一項。結果:

| tick | 淨 bits | recall@10 | 相對 clean |
|---|---|---|---|
| (clean) | 0 | 0.98376 | 100% |
| **0** | **6** | **0.42309** | **43%(已 collapse)** |
| 1 | 7 | 0.33837 | 34% |
| 2 | 13 | 0.19520 | 20% |
| 3 | 16 | 0.15700 | 16% |
| 5 | 24 | 0.08110 | 8.2% |
| 9 | 31 | 0.04546 | 4.6% |
| 20+ | ≥50 | ≤0.0103(平均 0.0009) | ≈0.1% |
| 99 | 165 | 0.00002 | ≈0 |

- **時間到懸崖 = 0 個 tick**(第一次量測即越過 collapse 線 recall < 0.4919)。
  6 bits 已足夠;E1 單 bit 圖(rotation 各 stage 平均 ΔRecall@10 0.167–0.200、
  100% catastrophic)與 E3b(16-bit 預算 → collapse_frac 1.0、ΔR≈0.81–0.82)
  早已指向這裡,本 run 把中間的完整軌跡補齊。
- **多 bit 效應是飽和型、非加法型**:6 bits 的 ΔR=0.56,遠小於 6×單 bit 平均(≈1.08,封頂),
  但 30 bits 時 ΔR 已 0.93 —— rotation 是全域共享的解碼基底,每一 bit 都讓「所有」向量的
  1-bit 估計失真,傷害快速趨飽和。
- **失效模式 100% silent-wrong**:100 ticks 無 crash、無 nan-inf。這就是「沉默崩潰」的
  完整實證 —— 沒有偵測層時,服務指標(QPS、錯誤率)完全正常,只有答案錯了。
- 尾端(tick 97–99)recall 1–2×10⁻⁵ ≈ 隨機猜測水準:rotation 三分之一損壞後,
  估計距離與真距離已近乎無關。

**與 slope 的對照**(參 expb):ex_code 同樣被 CRC 保護,但其單 bit 傷害 ~3×10⁻⁷
(E1),與 rotation 差 6 個數量級 —— 「懸崖 vs 斜坡」的命名由此而來。

---

## 4. E5 兩層防護:baseline(run 3)

### 4.1 主結果

- **recall@10 在 100 個 tick 全部恰等於 0.98376**(distinct 值只有一個)——
  與乾淨基線逐 tick 相等,防護下腐蝕對答案零影響。
- 最終 counters:
  `cliff_checked=51200`(512 bits × 100 ticks)、`cliff_repaired=235`、
  `cliff_irrecoverable=0`、`slope_failed=0`、`slope_reloaded=0`、`known_corrupted=0`、
  `oob_elements=0`、`eb_fraction=0`、`_eb_path` 全程 False。

### 4.2 三個內部一致性檢核(都通過)

1. **修復數 == 注入數**:cliff_repaired 累計 235 == run 1 的原始注入總數 235
   (同 seed 流)。逐 tick 修復平均 2.35 == 逐 tick 注入平均 —— 每 tick 的新損傷
   在當 tick 就被多數決全數修掉,**損傷從不跨 tick 累積**。
2. **XOR 互消在防護下消失**:run 1 有 70 bits 因重複命中互消,run 3 修復數 = 原始數
   (無互消)—— 因為每 tick 都修回 clean,不存在「已翻轉的 bit 再被翻回」的機會。
   這反過來印證了 run 1 淨值曲線的奇偶解讀。
3. **控制隔離乾淨**:只注入 rotation → slope(ex)與 bounds-check(指標)兩層
   全程零觸發。層與層之間無串擾。

### 4.3 壽命與成本

- 無防護:**tick 0 崩**;防護:**100 ticks 全平**。壽命增益下界 **>100×**
  (實驗長度所限;原記錄寫 >50×,本輪 run 2 的實測把分母收緊到 <1 tick,增益上修)。
- 成本:R=3 副本 = 192 B 記憶體(vs 索引 280 MB,佔 7×10⁻⁷)+ 每 tick 512-bit
  逐 bit 投票 + 一次 64 B CRC。這是 F3 成本模型(mem_cost + scrub_overhead)裡
  幾乎免費的一側 —— 「保護 64 B 就擋掉最大的災難」是量化索引防護的第一優先。

### 4.4 誠實 caveat(當時已標注,本輪 run 4 專門處理)

baseline 的 R=3 副本活在獨立記憶體、從未被注入 → `cliff_irrecoverable=0` 是**結構性**的
(不可能發生),不能當成多數決穩健性的證據。

---

## 5. 複本同時累積:多數決的失效邊界(run 4,新量測)

複本以**同一 per-bit 過程(p=0.005)、獨立流**(每複本一條 seed lane)每 tick 注入;
主 buf 照常注入。結果:

| tick | recall@10 | irrecoverable(累計) | 複本淨損 bits | 說明 |
|---|---|---|---|---|
| 0–1 | 0.98376 | 0 | [0,0,0](修後) | 投票成功,副本被同步回 clean |
| **2** | **0.36070** | **1** | [1,3,5] | **首次多數決失敗** |
| 3 | 0.27657 | 2 | [4,5,6] | 永久失效開始 |
| 5 | 0.14694 | 4 | [10,12,13] | |
| 99 | **0.00000** | **98** | [141,174,153] | 沿無防護軌跡崩到底 |

- `cliff_repaired` 凍結在 20(tick 1 之後再無成功修復);tick 2 起 **98/98 ticks 全部
  irrecoverable**;recall 軌跡與 run 2(無防護)幾乎重合(0.36→0.28→…→0)。
- **機制**:多數決失敗(某 bit 有 ≥2/3 副本同時錯)→ 依「不服務錯值」語意拒絕回寫
  → 副本不再被同步 → 副本損傷持續累積 → 之後每 tick 必然再失敗。
  **單次瞬時雙擊 = 永久解除防護**;這個設計是誠實的(絕不拿錯的 majority 蓋掉資料),
  但也是一次性的保險絲。
- **統計注**:tick 2 即失敗是偏早的單 seed 抽樣(每 tick 失敗機率隨副本淨損上升,
  以 tick 2 的 [1,3,5] bits 估計單 tick 約 4–5%,期望首失敗在 O(10) ticks 級);
  但「短命」的結論對這個量級不敏感 —— 同速累積下 R=3 多數決撐不過幾十個 tick。
  精確首失敗分布需多 seed 重複,列入後續。
- **設計含義**:多數決層要嘛(a)副本本身需要定期「對 clean CRC 驗證 + 重載」
  (等同 slope 層的 lazy reload 思路,把副本也納入 scrub 對象),要嘛(b)換成
  可糾錯編碼(ECC/parity 跨副本)。「複製了就安全」在同故障率假設下不成立 ——
  這正是本 run 要堵的 caveat,現在有數了。

---

## 6. 綜合圖景:懸崖—斜坡—恢復

| | rotation(懸崖) | ex_code(斜坡,參 expb) |
|---|---|---|
| 大小 | 64 B(全域一份) | 96 B × 10⁶(佔索引主體) |
| 單 bit 傷害(E1) | ΔR 0.17–0.20,100% catastrophic | ~3×10⁻⁷,100% benign |
| 無恢復時間線 | **tick 0 崩**(本文 run 2) | 輕損傷下幾乎平(expb `none`) |
| 對策 | R=3 多數決(本文 run 3/4) | CRC + drop / EB-fallback(expb) |
| 對策效果 | 100 ticks 全平;但複本同損時一次性失效 | EB−drop = +0.003@5% → +0.083@57% |
| 剩餘弱點 | 副本無自身 scrub → 一次雙擊永久失效 | `none` 僅在損傷重時才輸(嚴重度軸) |

F3 的敘事鏈因此完整:**懸崖端**用近乎免費的 192 B 複本把「<1 tick 靜默全崩」變成
「≥100 ticks 無損」(但要補副本 scrub);**斜坡端**用 CRC+EB 在偵測到的損壞上買到
與 drop 相比逐步放大的 recall 餘裕。兩者成本相加仍遠小於全 fp32 冗餘。

## 7. 限制與後續

1. 本文四組 run 都是 `uniform_accum × rotation` 單組合、單 seed(1234);
   E3c 其餘三種 pattern 對 rotation 的時間線、以及多 seed 的首失敗分布,尚未跑
   (expb 已示範 4 patterns 在 element 層的等價性,但 rotation 是 bit 層,不可直接外推)。
2. run 4 的複本 seed lane 沿用 `seed ^ tick ^ ((r+1)·0x5EED)`;expb 曾在類似 XOR lane
   上抓到 aliasing(已改 SeedSequence)。此處檢查過 tick/replica 視窗內無碰撞,
   但多 seed 重跑時建議一併遷移到 SeedSequence。
3. `--measure-recall` 每 tick 一次完整查詢(~15 s),100 ticks ≈ 27 分鐘;
   多 pattern × 多 seed 的矩陣建議 tmux 分批。
4. 副本 scrub(§5 設計含義 (a))是下一個最便宜、最有回報的機制實驗:
   在 `RecoveryGuard._scrub_cliff` 成功路徑外,對副本本身加 CRC 驗證 + 從 clean 重載。

## 附錄:重現指令與檔案

```bash
# run 1(注入動力學;決定性,可隨時重生)
python phase3_e3c_temporal.py --adapter real --region rotation
# run 2(無恢復 recall 時間線)
python phase3_e3c_temporal.py --adapter real --region rotation --measure-recall
# run 3(兩層防護 baseline)
python phase3_e5_recovery.py --adapter real --region rotation
# run 4(複本同時累積)
python phase3_e5_recovery.py --adapter real --region rotation --inject-replicas
```

產物(均在 `artifacts/phase3/`):`e3c/e3c_uniform_accum_rotation[_recall].{records.jsonl,json}`、
`e5/e5_uniform_accum_rotation[_replicas].{records.jsonl,json}` 與對應 run log。
交叉參照:`e1/vuln_map.csv`(單 bit 圖)、`e3b/e3b.json`(空間模式)、
`expb/`(斜坡端 EB 掃描)、`plan/stage2_plan1.md`(實驗 A 原始記錄)。
