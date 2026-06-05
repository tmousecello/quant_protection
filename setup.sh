#!/usr/bin/env bash
#
# quant_protection 環境建立腳本（量化索引 bit-flip 敏感度研究）
#
# 一鍵：建立 venv (系統 Python 3.13) -> 裝齊 FAISS + hnswlib 等依賴
#       -> 驗證 SIFT1M 就緒 -> 跑 verify_env.py smoke test。
# 冪等：可重複執行，.venv 已存在則沿用。
#
# 使用者決策：FAISS + hnswlib (pip wheel，無需 clone/編譯)、只用現有 SIFT1M、系統 Python 3.13。
#
# 用法:  bash setup.sh
#
set -euo pipefail

# 腳本所在目錄（不寫死絕對路徑）
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORK_DIR"

VENV_DIR="$WORK_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> WORK_DIR: $WORK_DIR"
echo "==> 使用 Python: $("$PYTHON_BIN" --version 2>&1) ($(command -v "$PYTHON_BIN"))"

# 1) 建立 / 沿用 venv ---------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
  echo "==> 已存在 venv，沿用: $VENV_DIR"
else
  echo "==> 建立 venv: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 2) 升級基礎工具 + 安裝依賴 --------------------------------------------------
echo "==> 升級 pip / wheel / setuptools"
python -m pip install --upgrade pip wheel setuptools

echo "==> 安裝依賴 (requirements.txt)"
python -m pip install -r "$WORK_DIR/requirements.txt"

# 3) 驗證 SIFT1M 就緒（本研究只用現有 SIFT1M）--------------------------------
echo "==> 檢查 SIFT1M 資料集"
missing=0
for f in sift/sift_base.fvecs sift/sift_query.fvecs sift/sift_groundtruth.ivecs; do
  if [ -f "$WORK_DIR/$f" ]; then
    echo "    [ok] $f"
  else
    echo "    [缺] $f" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "ERROR: SIFT1M 檔案不齊。本研究預設只用 sift/ 下的 SIFT1M，請先放置上述檔案。" >&2
  exit 1
fi
# 註：日後若要加 GIST1M (960d) 或文字 embedding (~768d)，於此處增列下載並更新 verify_env.py。

# 4) smoke test ---------------------------------------------------------------
echo "==> 執行環境驗證 (verify_env.py)"
python "$WORK_DIR/verify_env.py"

# 5) 完成提示 -----------------------------------------------------------------
cat <<EOF

==========================================================
 環境就緒。啟用方式：
   cd "$WORK_DIR"
   source .venv/bin/activate
==========================================================
EOF
