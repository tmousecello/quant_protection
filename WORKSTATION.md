# 在工作站上執行(搬機與重跑指南)

筆電上 Phase 1 全跑約 9.8h(序列,瓶頸 IVF_SQ8 的 sq_scale 窮舉),改到工作站平行跑約 5.3h。
程式碼與小型結果走 git;大型物(資料集 550M、索引 2.2G、venv)不入庫,在工作站重建。

## 一、版本控制(筆電端)

```bash
cd quant_protection
bash git_tasks.sh init               # 一次性:init + 初始 commit + 建私有 GitHub repo + push
bash git_tasks.sh save "改了 XXX"    # 之後每次存檔:add + commit + push(省略訊息用時間戳)
bash git_tasks.sh status             # 看狀態 / 最近 commit / 遠端
```

`git_tasks.sh` 有 **>50MB 大檔護欄**:若不小心 `git add` 到 `sift/` 或 `artifacts/indexes/`
之類大檔會自動中止並取消暫存,避免污染 repo。

## 二、工作站 bootstrap(從零自助)

```bash
git clone https://github.com/tmousecello/quant_protection.git
cd quant_protection

bash fetch_sift.sh          # 下載 SIFT1M 公開資料集 → sift/(冪等)
bash setup.sh              # 建 .venv(系統 python3)+ 裝 faiss/hnswlib + verify_env
source .venv/bin/activate

python phase0_build.py     # 重建 7 個索引 + ground truth + regions(約 5–15 分)
```

> 工作站若是 Linux,`setup.sh` 預設用系統 `python3`;FAISS 走 pip wheel,無需編譯。

> **乾淨起跑(重要)**:`phase0_build.py` 會在工作站**重建**索引並重生 region 地圖(含本輪
> 新增的 `codes_meta` 切割),所以 git 內的 region 地圖只是參考、會被覆蓋成與工作站索引相符的版本。
> 同理,repo 內committed 的 `artifacts/phase1/vuln_map.csv` 是**舊幾何、缺 IVF_SQ8**,會被下面
> §三 的完整 Phase 1 全跑覆蓋。無需手動處理。

## 三、Phase 1 平行跑(約 5.3h)

IVF_SQ8 自己約 5.3h,其餘四個合計約 4.5h;拆成兩條程序平行即可填滿 CPU。
Linux 無 `caffeinate`,用 `tmux`(建議)或 `nohup` 讓它在背景續跑:

```bash
source .venv/bin/activate
python phase1_region_accounting.py     # Task A,先補上 HNSW_SQ8 codes tail 的 regions_aug

# 兩條背景程序,各自獨立 shard、--resume 可斷點續跑
nohup python phase1_sensitivity.py --indexes IVF_SQ8 --resume > ivfsq8.log 2>&1 &
nohup python phase1_sensitivity.py --indexes FLAT,IVF_FLAT,HNSW,HNSW_SQ8 --resume > others.log 2>&1 &
wait

python phase1_sensitivity.py --aggregate-only   # 合併兩條的 shard → 完整 vuln_map
python phase1_report.py                          # C1 表 + C2/C3 圖
```

先驗證流程再開長跑:`python phase1_sensitivity.py --smoke`(約 2 分,跑 D1–D4 護欄)。

## 四、Phase 2(接在完整 Phase 1 之後)

**前置(乾淨起跑)**:Tier 3/Tier 4 吃 Phase 1 的 `vuln_map.csv` 與 `phase1/raw/*.records.jsonl`,
必須先完成上面 §三 的**完整** Phase 1 全跑——新幾何(`codes_meta` 切割)、且**含 IVF_SQ8**
(其 `sq_scale` 是主軸的災難結構)。Tier 1/Tier 2/burst 各自產資料,只需 §二的 `phase0_build.py`
+ `phase1_region_accounting.py` 重生的 `regions_aug`。

```bash
source .venv/bin/activate
# step0: 補 IVF_SQ8
python phase1_sensitivity.py --aggregate-only

# Tier 1(PQ 敏感度)與 Tier 2(graph_edges)互相獨立,可平行;皆 --resume 可斷點續跑
nohup python phase2_pq_sensitivity.py    --resume > t1.log 2>&1 &
nohup python phase2_graph_sensitivity.py --resume > t2.log 2>&1 &
wait

python phase2_rollup.py        # Tier 3:faults/MB 兩條曲線(主曲線排除 graph_edges,另列獨立曲線)
python phase2_detection.py     # Tier 4:值域護欄涵蓋率

# burst 子研究(獨立,任意時間)
python phase2_burst.py --trials 50
python phase2_burst.py --aligned-worstcase --trials 50
```

開長跑前每支先 `--smoke`(秒級)驗證 pipeline。**完整指令/參數/成本/輸出 schema、各 tier 的
乾淨起跑前置,見 `artifacts/phase2/README.md`(authoritative design 在 `prompt.md`)**。

## 五、把結果帶回筆電

```bash
# 工作站端
bash git_tasks.sh save "phase1 full results from workstation"
# 筆電端
bash git_tasks.sh pull
```

回到筆電的是**數值結果**:`artifacts/region_accounting.*`、`artifacts/phase1/vuln_map.*`、
`artifacts/phase1/sensitivity_table.*`、`artifacts/phase1/failure_modes.csv`;以及 Phase 2 的
`artifacts/phase2/vuln_map_pq.csv`、`graph_edges_characterization.csv`、
`rollup/curve{A,B}*.csv`+`region_terms.csv`、`detection/{coverage.csv,*summary.json,expected_ranges.json}`、
`burst/*_burst.csv`。

**注意**:`artifacts/phase1/raw/`、`artifacts/phase2/{raw/,_substrate/,**/*.npy,**/*.jsonl}`
(逐筆 jsonl + isolated worker 暫存 buffer)與 `artifacts/phase1/charts/`(png)已 git-ignore,
**留在工作站**。要在筆電看圖,就在工作站直接看,或自行 `scp` 圖檔/raw 回來
(raw 在筆電也能 `python phase1_report.py` 重畫圖)。
