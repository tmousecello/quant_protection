# 審計後修正包:fuse 線重跑指示 + 全文替換清單

> 兩部分:A. 重跑指示(選項 (a),對凍結開一個記錄在案的例外);B. 文字替換清單(逐文件、逐句:原文 → 改為)。全部完成後,審計發現的兩個 ARTIFACT 與三個定性問題即關閉。

---

## A. Fuse 線重跑指示(給 Claude Code / 可直接人工執行)

### 凍結例外聲明(寫進 plan 與 commit message)
> 凍結原則補充:不開新實驗;**已凍結結果中發現的混淆,允許以最小等價重跑修正,並記錄在案。**本次例外:主圖 fuse 線(run 4)為舊 XOR lane 產物,與 self-scrub 線(新 SeedSequence lane)相差兩個變數;重跑 fuse 線於新 lane,使主圖四線同 lane 同管線,配對對比恢復有效。審計出處:第二輪誠實性審計,發現一。

### 重跑規格(零新機制、零新參數)
- 腳本:`phase3_e5_recovery.py --adapter real --region rotation --inject-replicas`,**cliff-scrub 關閉**(legacy fuse 語意)、**帶 recall**。
- lane:現行 SeedSequence(不動碼)。
- **seed 選擇**:用 #2 分布中首失敗最接近 median(21)的 seed(從 `first_fail_summary.json` 查,如首失敗 tick ∈ [19,23] 者取其一;記錄選擇規則於 provenance)。理由:主圖的 fuse 線應呈現「典型」失效時點,不是最早或最晚的極端。
- ticks:200(與 self-scrub 線同長,圖上可對齊)。
- 預期:首失敗 ≈ 該 seed 在 #2 的 tick(同 lane 決定性,應完全一致——這本身是一個 sanity:**若首失敗 tick 與 #2 記錄不符,停,查 lane/參數**);失敗後 recall 沿無防護軌跡崩落。
- 產物:`e5/e5_uniform_accum_rotation_replicas_fuse_newlane.{json,records.jsonl}`,provenance 蓋 lane=SeedSequence、seed、選擇規則、審計例外引用。
- 主圖重生:`phase3_f_synth.py` 的 fuse 線輸入指向新檔;**並修 `phase3_f_synth.py:239-248`,加 lane 檢查**(讀輸入 records 的 provenance,四線 lane 不一致即 abort)——把審計抓到的「只檢查檔案存在」升級為「檢查 lane 一致」,防止回歸。
- 時間:~55 min(200 ticks 帶 recall)+ 圖重生秒級。

### 順帶(同一次工作站時段,可選)
- 無:此例外僅覆蓋 fuse 線。不要夾帶其他重跑。

---

## B. 全文替換清單

### B0. 措辭校準表(依審計證據強度分級;所有文件遵循)

| 主張 | 可用措辭 | 禁用措辭 |
|---|---|---|
| #1 run 內 200-tick 全平、4 次再生 | we show / 直接量測 | — |
| #1 「無限續命」 | 「機制上可再生;200 ticks 內 4 次投票失敗全數由 256 B 重載復原」 | 「無限」「indefinitely」(留給讀者推) |
| #1 fuse vs scrub 配對對比 | 重跑完成前:不用「唯一差異是 scrub」;完成後恢復 | (重跑前)「the only difference is...」 |
| #2 median 21 分布 | we measure;「與先驗解析模型**量級一致**(median 17.7 vs 21)」 | 「吻合」「match」「agree well」 |
| #3 G 對角線 | 「構造驗證(by construction):偵測層在服務路徑上即產出決策變數 f」 | 「偵測準確率」「逐點相等」當成實驗發現 |
| anchor 19/0 | 「設計完備性:堵理論盲區;stub 證明邏輯上會抓;本故障模型下按構造不會觸發」 | 「保底運轉正常/未抓到事件」暗示其在場上待命 |
| irrecoverable=0 | 從證據列表**移除**;若提及:「該路徑在 scrub 語意下按構造不可達」 | 當作 gate/證據引用 |

### B1. `STAGE3_ANALYSIS.md`

1. **TL;DR 第 2 點**
   - 原:「200 ticks recall 全程恰等於 0.98376…`irrecoverable` 恆 0。…壽命增益下界…上修為 >200×。」
   - 改:「200 ticks recall 全程恰等於 0.98376(= clean),期間 4 次多數決失敗(tick 77/79/94/137)全數由觸發式全量重載復原(`cliff_vote_fail_reloads=4`、`cliff_repaired=2012`)——保險絲情境真實發生且每次再生;機制上可再生,200 ticks 為實測下界。(註:`cliff_irrecoverable` 在 scrub 語意下按構造不可達,不列為證據。)」
2. **TL;DR 第 3 點**:「同量級吻合」→「量級一致(median 17.7 vs 21,差 19%)」。
3. **TL;DR 第 5 點**
   - 原:「觀察比例與注入真值完全相等(max |dev|=0)——偵測層直接吐出決策變數 f,§3.3 證據成立。」
   - 改:「構造驗證(by construction,分子由 run 內斷言釘死):偵測層在正常服務路徑上零額外計算地產出決策變數 f,可直接對 f\* 表排程;此為設計性質的驗證,非偵測準確率的實驗發現。」
4. **§2 表 Phase A gate 行**:刪「`cliff_irrecoverable=0`」作為 gate 證據;加註「(audit:該 gate 為套套邏輯,已自證據鏈移除;真實證據為 reloads=4 / recall 全平 / repaired=2012)」。
5. **§3 表 anchor 行**
   - 原:「低頻保底運轉正常,未抓到『同位同值雙擊』盲區事件」
   - 改:「anchor 檢查 19 次、0 mismatch:本故障模型下盲區按構造不可觸發(注入器不及記憶體內 anchor;觸發需 CRC-32 碰撞 ~2⁻³²);anchor 的價值為設計完備性,stub 測試證明其邏輯有效」。
6. **§3 對比 run 4 段 + 壽命敘事段**:所有「run 4 tick 2」改為「舊 lane 早期觀察(run 4);無偏 lane 下首失敗 median 21(N=30,§4)」;主圖說明加「fuse 線已於新 lane 重跑(檔名 _fuse_newlane),四線同 lane」【重跑完成後】。
7. **§7 完成定義表 #4 行**:「✅ 80 點逐點相等」→「✅ 構造驗證:服務路徑即產出 f(80 點,by construction)」。
8. **§8 限制**:新增第 6 條:「主圖 fuse 線初版為舊 XOR lane 產物(審計發現),已依凍結例外於 SeedSequence lane 重跑替換;run 4 保留為歷史記錄,不再被主圖或壽命敘事引用。」

### B2. `E3C_E5_ANALYSIS.md`

1. **TL;DR 第 3 點與 §5 全節**:標註「本節數字為舊 XOR lane(aliasing-prone);分布層結論由 Stage 3 #2(新 lane,median 21、IQR [9,30]、min 0)取代;本節保留為機制發現的歷史記錄——『多數決是一次性保險絲』的機制敘述不受 lane 影響(新 lane 兩條極端 seed timeline 重現同一機制)。」
2. **§5 統計注**:「期望首失敗在 O(10) ticks 級」後加「(Stage 3 實測 median 21,確認)」;「tick 2 即失敗是偏早的單 seed 抽樣」後加「(且受舊 lane aliasing 加成;新 lane 分布見 Stage 3)」。
3. **§7.3 (c) 與 §7.4**:引用保險絲處的「tick 2」全部改「median ~21 ticks(N=30)」。

### B3. `paper_draft_zh.md`

1. **Abstract 第三句**
   - 原:「…以複本多數決搭配自我 scrub 修復(並實測證明『複製而不 scrub 只是一次性保險絲』)…整體保護開銷僅為索引大小的 0.0001%…」
   - 改:「…以複本多數決搭配自我 scrub 修復——我們實測證明,在錯誤反覆發生的記憶體上,不自我 scrub 的複本是一次性保險絲(首次失效 median 僅 ~21 個錯誤週期,N=30),而事件觸發的 256 B 重載使其可再生;對佔索引 99.9% 的碼資料,以 CRC 偵測搭配 error-bound-aware 降級檢索容忍錯誤累積——災難防護僅需 132 B(4.7×10⁻⁵%),偵測層開銷依 CRC 粒度為 0.03–5.7% 且可解析地權衡。」【同時解決前輪指出的 0.0001% 與 5.7% 矛盾】
2. **Abstract 第四句**:「以 192 B 的複本成本將壽命延長逾 100 倍」→「以 132 B 的複本成本使防護可再生(200 個錯誤週期內 4 次失效全數復原)」。
3. **Intro 第四段**:「實測將『6 位元即沉默全崩』延長為逾百個錯誤週期無損,成本 192 B;我們並實測揭露…一次瞬時雙擊即永久失效」→「實測顯示無防護時 6 位元即沉默全崩;複本多數決可修復,但若複本自身不被 scrub,在反覆錯誤下為一次性保險絲(首次失效 median ~21 週期);事件觸發的全量重載(自持久 clean 源,256 B)使防護可再生——200 週期內 4 次失效全數復原,recall 全程無損」。
4. **§3.2**:「(tick 2 失效、其後 98/98 週期不可修復)」→「(N=30 分布:median 21、IQR [9,30]、最早 tick 0;失效後沿無防護軌跡崩落)」。
5. **§4 分析**:間隔比表前加一句:「解析模型之三項輸入(p、結構位元數、R)均為先驗物理量,與量測分布量級一致(median 17.7 vs 21)。」;新增 CRC 粒度小段:「偵測開銷隨 CRC 粒度可解析權衡:每元素 16 B manifest 為 5.7%,CRC32/元素為 1.4%,4 KB page 級為 ~0.03% 但放大 fallback 比例 ~43×、等比縮短安全重載間隔——低錯誤率下 f_page ≈ 43·f_elem,粒度甜蜜點為 p 之函數。」【即被砍實驗 D 的解析版】
6. **§5.2 主結果句**:「複本不 scrub 的保險絲失效(tick 2 起…)」→「複本不 scrub 的保險絲失效(典型 seed,首失敗 ~median)」【重跑完成後填實際 tick】;圖說明加「四線同 SeedSequence lane、同注入管線,唯一差異為防護配置」。
7. **§5.2 supporting**:G 的引用照 B0 措辭(「構造驗證」);anchor 若提及照 B0。
8. **Conclusion**:壽命與保險絲數字同步(median 21、可再生、132 B)。

### B4. `progress` / advisor 溝通文件(若引用)
- 所有「>100×/>200× 壽命」的口徑統一為:「無防護 tick-0 崩;自我 scrub 下 200 週期全平且可再生」;所有「tick-2」→「median 21」。

### B5. 流程資產(新增,一小節或附錄)
- 將兩輪審計(第一輪 8/8 CLEAN 假陰性 + 第二輪翻案)存檔為 `audits/`;論文或 artifact 說明可加一句:「所有 gate 經對抗性審計覆核;構造性 gate(不可能 fail 者)已自證據鏈移除。」
- 工程備忘(不進論文):`phase3_f_synth.py` 加 lane 一致性檢查(A 節已含);gate 設計守則——**答不出 fail 條件的 gate 不是 gate**(本專案已三次出現構造性成功:run 3 複本不朽、irrecoverable gate、G 對角線)。

---

## 執行順序
1. A 節重跑(~1 h 工作站)→ 主圖重生 → B1.6 / B3.6 填實際 seed 與首失敗 tick。
2. B1–B4 文字替換(開發機,半天內)。
3. B5 存檔與 f_synth lane 檢查(順手)。
4. 完成後:審計兩發現關閉,證據—措辭對齊,回到寫作主線(下一站:成本敘述已在 B3.1/B3.5 一併修正,可直接續寫第三、四章的肉)。