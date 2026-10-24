#!/usr/bin/env bash
# Wrapper used by the systemd timers: sync code, run one job, publish logs back to GitHub.
# usage: deploy/run.sh live|daily
set -uo pipefail
cd "$(dirname "$0")/.."
exec 9>/tmp/tkiller.lock
flock 9
set -a; source .env; set +a
git pull -q --rebase --autostash origin main || echo "git pull failed"
.venv/bin/python -m "tk.$1"
status=$?
git add data/live data/plan results LATEST.md 2>/dev/null
if ! git diff --cached --quiet; then
  git -c user.name="t-killer-bot" -c user.email="t-killer-bot@users.noreply.github.com" commit -q -m "$1 $(date -u +%F' '%H:%M)"
  git push -q origin main || { git pull -q --rebase origin main && git push -q origin main; } || echo "git push failed"
fi
exit $status
