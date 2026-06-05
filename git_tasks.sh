#!/usr/bin/env bash
#
# git_tasks.sh — quant_protection 的 git 任務自動化(筆電 / 工作站共用)
#
#   bash git_tasks.sh init           # 一次性:git init + 初始 commit + 建私有 GitHub repo + push
#   bash git_tasks.sh save ["訊息"]  # 持續版本追蹤:add -A + commit(+ push);省略訊息則用時間戳
#   bash git_tasks.sh status         # 工作目錄狀態 + 最近 commit + 遠端
#   bash git_tasks.sh pull           # 工作站側:fast-forward 拉取
#
# 安全機制:任何 add 後若暫存到 >50MB 的檔(疑似誤加 sift/ 或 artifacts/indexes/),
#           一律中止並取消暫存,避免把大型二進位 commit 進 git。
#
# 遠端可用環境變數覆蓋:GIT_TASKS_REPO=<owner/name>(預設 tmousecello/quant_protection)
#
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 切到 repo 根(find 路徑需相對於此)

REPO="${GIT_TASKS_REPO:-tmousecello/quant_protection}"
MAXSIZE_MB=50

# --- 大檔護欄 ----------------------------------------------------------------
check_large() {
  local big
  big=$(git diff --cached --name-only -z | while IFS= read -r -d '' f; do
    [ -f "$f" ] && find "$f" -size +${MAXSIZE_MB}M -print
  done)
  if [ -n "$big" ]; then
    echo "ABORT: 偵測到 >${MAXSIZE_MB}MB 的暫存檔,疑似誤加大型二進位:" >&2
    echo "$big" | sed 's/^/   /' >&2
    echo "請把它們加進 .gitignore(若確定要入庫,改用 git add -f)。已取消本次暫存。" >&2
    git reset -q
    return 1
  fi
}

# --- init --------------------------------------------------------------------
cmd_init() {
  if [ -d .git ]; then
    echo "已是 git repo;init 跳過 git init(請改用 save)。"
  else
    git init -b main >/dev/null
    echo "已初始化 git repo(branch=main)。"
  fi
  [ -f .gitignore ] || echo "[warn] 找不到 .gitignore — 建議先建立,否則大檔可能被加入。"

  git add -A
  check_large
  if git diff --cached --quiet; then
    echo "沒有可提交的變更。"
  else
    git commit -q -m "initial commit: quant_protection Phase 0/1 code + small results"
    echo "已建立初始 commit。"
  fi

  if git remote get-url origin >/dev/null 2>&1; then
    git push -u origin main
  elif command -v gh >/dev/null 2>&1 && gh repo view "$REPO" >/dev/null 2>&1; then
    git remote add origin "https://github.com/$REPO.git"
    git push -u origin main
    echo "已連到既有遠端 $REPO 並推上 main。"
  elif command -v gh >/dev/null 2>&1; then
    gh repo create "$REPO" --private --source=. --remote=origin --push
    echo "已建立私有 GitHub repo $REPO 並推上 main。"
  else
    echo "未安裝 gh CLI;請手動:git remote add origin <url> && git push -u origin main"
  fi
}

# --- save --------------------------------------------------------------------
cmd_save() {
  git add -A
  check_large
  if git diff --cached --quiet; then
    echo "沒有變更可提交。"
    return 0
  fi
  local msg="${1:-auto: $(date '+%Y-%m-%d %H:%M:%S')}"
  git commit -q -m "$msg"
  if git remote get-url origin >/dev/null 2>&1; then
    git push
    echo "已提交並推送: $msg"
  else
    echo "已提交(尚未設定 origin,略過 push): $msg"
  fi
}

# --- status ------------------------------------------------------------------
cmd_status() {
  echo "--- 工作目錄 ---"
  git status -s || true
  echo "--- 最近 commit ---"
  git --no-pager log --oneline -5 2>/dev/null || echo "(尚無 commit)"
  echo "--- 遠端 ---"
  git remote -v 2>/dev/null | head -2 || echo "(無遠端)"
}

# --- pull --------------------------------------------------------------------
cmd_pull() { git pull --ff-only; }

# --- dispatch ----------------------------------------------------------------
case "${1:-}" in
  init)   cmd_init ;;
  save)   shift || true; cmd_save "${1:-}" ;;
  status) cmd_status ;;
  pull)   cmd_pull ;;
  *)
    echo "用法: bash git_tasks.sh <init|save [\"訊息\"]|status|pull>" >&2
    exit 2 ;;
esac
