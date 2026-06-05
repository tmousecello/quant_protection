#!/usr/bin/env bash
# 已棄用：請改用 setup.sh（建立 venv + 裝 FAISS/hnswlib + 驗證環境）。
# 保留此檔僅為轉址。
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup.sh" "$@"
