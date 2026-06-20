# Claude Code 指示:Phase 3 — Stage 0(量測基礎建設)

> 這份是給你(Claude Code)的自含 brief。先讀「開始前」一節做探索並提計畫,再實作。Stage 0 只建**基礎建設 + 驗證**,**不要**跑 Stage 1 的大規模刻畫實驗。不要執行需要花費大量時間的主要計算工作，實作完成的驗證只跑 smoke run。

---

## 任務總覽
在 Samuel 的 RaBitQ 環境裡,建好三個量測元件——**fault model、recall 量測、cost 量測**——並為每個元件附上驗證。這些是後續所有 Phase 3 實驗的共同地基。核心原則:**量測/度量的邏輯一律 import 自既有的成熟核心,不重寫**(細節見「核心架構決定」)。

---

## 開始前:先讀與探索(請先做這步,再提 plan)
1. 讀根目錄 `CLAUDE.md`(專案慣例與背景)。
2. 探索兩個 repo 並回報你的理解:
   - `quant_protection/`(我的 repo):量測/刻畫核心。重點看 `qp/` 下的 `metrics.py`(recall、tolerant recall、`is_silent_collapse`、failure 分類)、`flip.py`(serialize→XOR→deserialize 注入)、`regions.py`、`config.py`(鎖定變數)、`buckets.py`。理解它現有的 fault 注入與度量介面。
   - `StorageSystemProject_RaBitQ_Free-recovery/`(Samuel 的 repo):RaBitQ 索引與恢復機制。重點找出:RaBitQ 索引怎麼建/載入/序列化、其位元組佈局(rotation matrix / factors / bin(1-bit) / ex(擴展位) / pointer 各在哪)、b=7 的乾淨 recall、以及恢復政策(EB-aware fallback)的碼。
3. **環境共存測試**:回報 `quant_protection` 的量測核心(尤其 `qp/metrics.py`、`qp/flip.py`)能否在 Samuel 的 env import 成功。若有 FAISS-only 硬依賴和 RaBitQ-library 衝突,標示出來——我們只需要純 Python/numpy 的度量與注入部分能 import。
4. 基於以上,提出 Stage 0 的實作 plan(照「建構順序」一節)。

---

## 專案背景(讓你理解為什麼這樣建)
- **研究主題**:向量資料庫 / ANN 索引在**會復發的記憶體位元錯誤**下的資料保護。核心發現:容錯風險高度不均——大宗 codes/vectors 幾乎免疫,危險集中在 KB 級的全域共享解碼結構(SQ8 的 `sq_scale` 是災難性單點;預期 RaBitQ 的**全域旋轉矩陣**是其對應的單點災難)。
- **保護範式**:criticality-aware 不對稱保護(UEP)+ 利用 RaBitQ 多精度層(bin/ex)的**免費記憶體內恢復**(CRC 觸發 → fallback 到已常駐的低精度 bin + error-bound 悲觀排序,不重載硬碟)。因錯誤會復發 → 採三層 scrubbing。
- **威脅模型**:記憶體錯誤、**會復發**(修好的 bit 會再壞)→ 腐蝕隨時間累積 → 需 scrubbing。三個 fault model 見下節。
- **目標場**:CIDR(願景論文)。Stage 0 是地基;Stage 1 之後才是 RaBitQ 刻畫、恢復機制、scrub-interval 評估。
- **關鍵度量語意**:`is_silent_collapse` = 「沉默崩潰」= 保留率 < 50% of own clean recall@10,且**只算沉默**(排除 crash **與 nan-inf**)。這是剛修正過的定義,務必用既有實作、勿自創。

---

## 目錄與兩個 repo 的角色
```
.
├── CLAUDE.md                                  # 先讀
├── StorageSystemProject_RaBitQ_Free-recovery  # Samuel:RaBitQ 索引 + 恢復機制(在此 env 工作)
├── first-try
├── paper
└── quant_protection                           # 我:qp/ 量測核心(度量/注入的單一來源,import 它)
```

---

## 核心架構決定(最重要的規則,違反會毀掉後續整合)
1. **在 Samuel 的 env 工作**(RaBitQ 在那邊成熟)。
2. **量測/度量核心一律 import 自 `quant_protection/qp/`,不要在 Samuel repo 重寫一份**。`is_silent_collapse`、recall@k、tolerant recall、fault 注入器(serialize→XOR→deserialize)必須是**同一份碼**。重寫會造成兩份實作悄悄分歧,讓之後跨 repo 的 parity check 失敗,並重新引入剛修掉的 bug。
3. 若 env 共存測試顯示無法直接 import(FAISS 衝突),則只把**純 Python/numpy 的度量與注入模組**抽成可 import 的小套件(submodule 或 `pip install -e`),FAISS-specific 的建索引部分留在 `quant_protection`。
4. RaBitQ 內部佈局(rotation/factors/bin/ex/pointer 的 offset/len)請**讀 Samuel 的碼找出來**,不要猜;若找不到,標示並回報,不要假設。

---

## 要建的東西(Stage 0 元件,各附 build + 驗證 + 驗收)

### 元件 0 · RaBitQ adapter + loaders + clean baseline(其餘都依賴它,先做)
- **build**:一個 adapter,給定 RaBitQ 索引 → 回傳 ① 位元組區域地圖(rotation / factors / bin / ex / pointer 的 offset、len)② serialize / deserialize。loaders:dataset、固定 query set、精確 ground truth。
- **驗證**:建乾淨 RaBitQ 索引,量 recall@10,應 ≈ **0.983**(b=7,對齊 Samuel 的既有結果)。對不上就停,先回報。
- **驗收**:adapter 能正確切出五個區域;clean baseline 數字對齊。

### 元件 1 · Fault model(三個模型,詳規見下節)
- **build**:統一注入介面,可指定 region、可設 seed、可還原;沿用 `qp/flip.py` 的 serialize→XOR→deserialize substrate。
- **驗證**:見下節每個模型的驗證項(bit-count、空間群聚、還原 roundtrip、決定性、獨立性、temporal 累積曲線)。
- **驗收**:三個模型各自的驗證單元測試通過。

### 元件 2 · Recall 量測
- **build**:`recall@{1,10,100}`、`tolerant_recall(ε)`、`is_silent_collapse`——**全部 import 自 `qp/`**(排除 crash 與 nan-inf)。固定 query/gt/k。在**實際搜尋路徑**上量(bin traversal + ex/refine rerank,且套用恢復政策),不要只重算解碼距離。
- **驗證**:
  - gt sanity:精確(brute-force)recall@1 對 shipped gt ≈ 1.0。
  - baseline anchor:clean b=7 ≈ 0.983。
  - 方向性 sanity:recall 隨 efSearch 單調升;`recall(clean) ≥ recall(corrupted)`;精度階梯 `1-bit≈0.41 < 4-bit≈0.889 < 7-bit≈0.983`。
  - 謂詞單元測試:`is_silent_collapse(0.4,0.95)=True`、`(0.89,0.95)=False`、`crash(None)=False`、`nan-inf=False`。
- **驗收**:以上全通過。

### 元件 3 · Cost 量測
- **build**:
  - 記憶體成本:`mem_cost = Σ_struct (mult-1)·struct_bytes + Σ checksum_bytes`(由元件 0 的 region map + 保護分配算)。
  - scrub 開銷:`scrub_cost = scrub_bytes × freq`(或表成佔 throughput 的 %);偵測(CRC/range-check)每查詢開銷。
  - F3 用統一成本:`total = mem_cost + scrub_overhead`。
  - 註:scrub 的**具體數值**要等 Stage 1 的 E1 確定「scrub 哪些結構、各多大」才填;Stage 0 先把**公式 + 對帳機制**建好。
- **驗證**:
  - 解析 vs 實測:解析算出的保護 bytes == 受保護索引實際序列化大小增量(assert ≈)。
  - scrub 時間實測:量小結構(如旋轉矩陣)scrub-repair 的時間,sanity 在微秒級。
  - 單調性:保護強度 ↑ → 成本 ↑。
- **驗收**:解析與實測對帳通過。

### 元件 4 · 整合驗證(Stage 0 的總驗收)
- **parity check(最關鍵)**:同一 clean RaBitQ 索引、同一 flip、同一 query/gt → 你 import 的核心與 Samuel 既有碼算出的 recall / collapse **完全相同**。
- **golden-file 回歸**:在一個小 smoke fixture 上把三元件的已知良好數字存成 golden,之後任何改動須重現。

---

## 三個 fault model(詳細規格)

> 全部在軟體層對 index 記憶體注入(不依賴 QEMU / 真硬體)。每個都要可指定 region、可設 seed、可還原。

**1. Uniform random multi-bit flip**
在一個 codeword 範圍內,每一個 bit 都有相同的、獨立的機率 p 被翻轉,p 的取值在 10^-6 到 10^-3 之間掃過幾個量級。對應 cosmic ray SEU、retention failure 等空間無相關的錯誤。
- 介面:`uniform_p(buf, region, p, seed)`,p ∈ `{1e-6 … 1e-3}`。
- 驗證:翻轉數 ≈ N·p(落在 binomial CI);位置直方圖 chi-square 不顯著偏離均勻(獨立性);同 seed → 相同位置(決定性);inject→restore 後 buffer 位元組完全相同。

**2. Spatially clustered multi-bit flip**
模擬 rowhammer 與 retention clustering 的結構化錯誤。Bit flip 不獨立,而是集中在一個 cache line 內某個固定大小 window 中(例如連續 8 bit 內出現 2~3 個 flip),或集中在物理上相鄰兩個 row 之間的對應位置。
- 介面:`spatial_cluster(buf, region, W, k, n_win, seed)`(W=window 大小如 8,k=每 window flip 數如 2~3);另一變體 `cross_row(buf, region, stride, seed)`(翻相鄰 row 對應位置)。
- 驗證:每 window 內翻轉數 == k;flips 確實群聚(非均勻散布)的 assert;決定性;還原 roundtrip。

**3. Temporally clustered burst flip**
模擬硬體老化與熱點的暫時性 error burst。時間軸上有一段「靜默期」幾乎無錯誤,然後在某觸發條件下短時間內出現一連串錯誤。
- 介面:`temporal_burst(buf, region, timeline, seed)`,timeline = 靜默期 + 觸發點的 burst(一串 N flips);需 stateful wrapper,腐蝕跨查詢累積(供 Stage 1 的 scrub-interval 實驗用)。
- 驗證:累積曲線(腐蝕比例 over time)符合模型(靜默 → 觸發點階躍);累積 bit 數隨時間如預期;同 seed → 相同時間序列。

---

## 規則與邊界
- **Do**:import `qp/` 度量核心當單一來源;讀 Samuel 的碼找 RaBitQ 佈局;每個元件先過驗證再進下一個;最後跑 parity check 當總驗收;每步小提交、可回溯。
- **Don't**:不要重寫 metrics / fault injector / collapse 謂詞;不要跑 Stage 1 的大規模 RaBitQ 刻畫或恢復評估(那是後續);不要猜 RaBitQ 內部佈局;不要手改任何產物檔(一律碼生成)。不要執行需要花費大量時間的主要計算工作。
- 不確定 RaBitQ 佈局或 env 共存細節時,**回報並停**,不要假設後硬幹。

---

## 建構順序(Stage 0 內部依賴)
1. 元件 0(adapter / region map / loaders / clean baseline)→ 先驗 b=7 ≈ 0.983。
2. 元件 1(fault model)與 元件 2(recall)可並行,各跑單元驗證。
3. 元件 3(cost,需 region map)。
4. 元件 4(parity check + golden-file)= Stage 0 總驗收。

**Stage 0 完成 = adapter 就緒 + 三元件各自驗證通過 + parity check PASS + golden 存檔。** 完成後回報結果,等待 Stage 1(E1:RaBitQ 脆弱度刻畫,含旋轉矩陣)的指示。