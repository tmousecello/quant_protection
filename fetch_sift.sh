#!/usr/bin/env bash
#
# fetch_sift.sh — 下載 SIFT1M 公開資料集到 sift/(不進 git,~168MB 壓縮 → ~550MB)。
# 冪等:檔案已就緒則秒退。來源同 ../first-try/build_hnswlib.sh 的 texmex 鏡像。
#
# 用法:  bash fetch_sift.sh
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
NEED=(sift/sift_base.fvecs sift/sift_query.fvecs sift/sift_groundtruth.ivecs sift/sift_learn.fvecs)

all_present() { for f in "${NEED[@]}"; do [ -f "$f" ] || return 1; done; }

if all_present; then
  echo "==> SIFT1M 已就緒,略過下載。"
  ls -la sift/
  exit 0
fi

tmp="sift.tar.gz"
echo "==> 下載 SIFT1M: $URL"
if command -v wget >/dev/null 2>&1; then
  wget -c "$URL" -O "$tmp"
elif command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "$tmp" "$URL"
else
  echo "ERROR: 需要 wget 或 curl 才能下載。" >&2
  exit 1
fi

echo "==> 解壓(解出 sift/ 目錄)"
tar xzf "$tmp"           # texmex 壓縮檔內含 sift/ 目錄,直接解到當前目錄
rm -f "$tmp"

if all_present; then
  echo "==> 完成。"
  ls -la sift/
else
  echo "ERROR: 解壓後仍缺檔,請確認壓縮檔結構:" >&2
  ls -la sift/ 2>/dev/null || true
  exit 1
fi
