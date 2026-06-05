#!/usr/bin/env python3
"""環境 smoke test：確認本研究的核心動作都能跑。

涵蓋：
  1. import + 版本回報 (numpy / faiss / hnswlib / scipy)
  2. 讀 SIFT1M 標頭確認 128 維
  3. FAISS IndexFlatL2 + IndexIVFFlat sanity search (k=10)
  4. bit-flip 地基：serialize_index -> 翻一個 bit -> deserialize_index 往返
  5. hnswlib binding sanity

任一步失敗即以非零碼結束。純讀取，不寫任何檔案。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIFT_BASE = os.path.join(HERE, "sift", "sift_base.fvecs")
DIM_EXPECTED = 128
N_SAMPLE = 10_000  # 只取前 N 條做 smoke test，避免載入整個 516MB


def step(msg: str) -> None:
    print(f"[check] {msg}", flush=True)


def read_fvecs(path: str, max_rows: int | None = None):
    """讀 .fvecs：每筆 = int32 維度 + dim 個 float32。回傳 (N, dim) float32。"""
    import numpy as np

    with open(path, "rb") as f:
        dim = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        f.seek(0)
        row_bytes = 4 + dim * 4
        count = max_rows * (row_bytes // 4) if max_rows is not None else -1
        raw = np.fromfile(f, dtype=np.float32, count=count)
    raw = raw.reshape(-1, dim + 1)
    return np.ascontiguousarray(raw[:, 1:]), dim


def main() -> int:
    # 1. imports + versions ---------------------------------------------------
    step("import numpy / faiss / hnswlib / scipy")
    import numpy as np
    import faiss
    import hnswlib
    import scipy

    print(f"        numpy   {np.__version__}")
    print(f"        faiss   {faiss.__version__}")
    print(f"        hnswlib {getattr(hnswlib, '__version__', '?')}")
    print(f"        scipy   {scipy.__version__}")

    # 2. SIFT header ----------------------------------------------------------
    step(f"讀 SIFT base 標頭: {SIFT_BASE}")
    if not os.path.exists(SIFT_BASE):
        print(f"        [FAIL] 找不到 {SIFT_BASE}", file=sys.stderr)
        return 1
    xb, dim = read_fvecs(SIFT_BASE, max_rows=N_SAMPLE)
    if dim != DIM_EXPECTED:
        print(f"        [FAIL] 維度 {dim} != {DIM_EXPECTED}", file=sys.stderr)
        return 1
    print(f"        dim={dim}, 取樣向量數={xb.shape[0]}")
    xq = xb[:5].copy()  # 拿前 5 條當查詢

    # 3. FAISS FlatL2 + IVFFlat ----------------------------------------------
    step("FAISS IndexFlatL2 sanity search (k=10)")
    flat = faiss.IndexFlatL2(dim)
    flat.add(xb)
    D, I = flat.search(xq, 10)
    assert I.shape == (5, 10) and (I[:, 0] == np.arange(5)).all(), "Flat 自查最近鄰應為自身"

    step("FAISS IndexIVFFlat sanity search (k=10)")
    nlist = 64
    quantizer = faiss.IndexFlatL2(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
    ivf.train(xb)
    ivf.add(xb)
    ivf.nprobe = 8
    D2, I2 = ivf.search(xq, 10)
    assert I2.shape == (5, 10), "IVF 查詢應回傳 (5,10)"

    # 4. bit-flip 地基：serialize -> flip -> deserialize ----------------------
    step("bit-flip 往返：serialize_index -> 翻 1 bit -> deserialize_index")
    buf = faiss.serialize_index(flat)          # numpy uint8 array
    buf = np.array(buf, dtype=np.uint8, copy=True)
    orig = buf.copy()
    # 翻動向量資料區(避開最前面的 header magic)的某個 bit
    pos = len(buf) // 2
    buf[pos] ^= np.uint8(1 << 3)
    assert not np.array_equal(buf, orig), "翻 bit 後 buffer 應改變"
    flipped = faiss.deserialize_index(buf)     # 往返成功代表方法可行
    Df, If = flipped.search(xq, 10)
    assert If.shape == (5, 10), "翻 bit 後的 index 仍應可查詢"
    print("        serialize/deserialize 往返 OK，corruption 注入路徑可用")

    # 5. hnswlib --------------------------------------------------------------
    step("hnswlib Index('l2', 128) 建立 + 查詢")
    h = hnswlib.Index(space="l2", dim=dim)
    h.init_index(max_elements=xb.shape[0], ef_construction=100, M=16)
    h.add_items(xb, np.arange(xb.shape[0]))
    h.set_ef(50)
    labels, dists = h.knn_query(xq, k=10)
    assert labels.shape == (5, 10) and (labels[:, 0] == np.arange(5)).all(), "hnswlib 自查最近鄰應為自身"

    print("\nENV OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
