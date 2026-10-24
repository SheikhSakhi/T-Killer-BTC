#!/usr/bin/env bash
# One-time server setup for T-Killer on Ubuntu 24.04 (Oracle Cloud, Frankfurt).
# Run as the ubuntu user:  bash setup.sh
# Secrets are typed here and stored only in /opt/tkiller/.env (readable by you alone).
set -euo pipefail
REPO=SheikhSakhi/T-Killer-BTC
DIR=/opt/tkiller

echo "== system packages"
sudo timedatectl set-timezone UTC
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git >/dev/null

read -rsp "GitHub fine-grained token (Contents: read/write on $REPO): " GH; echo
sudo mkdir -p "$DIR" && sudo chown "$USER" "$DIR"
if [ ! -d "$DIR/.git" ]; then
  git clone -q "https://x-access-token:${GH}@github.com/${REPO}.git" "$DIR"
fi
cd "$DIR"
chmod 700 .git

echo "== python environment"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-tk.txt

echo "== secrets (input hidden; press Enter to leave one empty for now)"
read -rsp "Telegram bot token: " TG; echo
read -rp  "Gmail address the bot sends from: " GU
read -rsp "Gmail app password (16 letters): " GP; echo
read -rp  "Email address that receives alerts [$GU]: " GT; GT=${GT:-$GU}
read -rsp "Binance TESTNET API key: " BK; echo
read -rsp "Binance TESTNET API secret: " BS; echo
GP=${GP// /}   # Google shows the app password with spaces; it works without them
umask 077
cat > .env <<ENV
MODE='paper'
BINANCE_TESTNET='1'
BINANCE_KEY='$BK'
BINANCE_SECRET='$BS'
TELEGRAM_TOKEN='$TG'
TELEGRAM_CHAT_ID=''
GMAIL_USER='$GU'
GMAIL_APP_PASSWORD='$GP'
ALERT_EMAIL_TO='$GT'
ENV
umask 022

if [ -n "$TG" ]; then
  echo "== Telegram: send /start to your bot now, then press Enter"; read -r
  CID=$(set -a; source .env; .venv/bin/python -m tk.notify chat-id | head -1)
  sed -i "s/^TELEGRAM_CHAT_ID=.*/TELEGRAM_CHAT_ID='$CID'/" .env
  echo "chat id: $CID"
fi

echo "== test alert"
(set -a; source .env; .venv/bin/python -m tk.notify) || true

echo "== downloading 2 years of market data (about 15 minutes)"
.venv/bin/python -m tk.data --months 24 | tail -3

echo "== first nightly run (builds tomorrow's plan)"
chmod +x deploy/run.sh
deploy/run.sh daily | tail -15 || true

echo "== timers"
sudo tee /etc/systemd/system/tk-live.service >/dev/null <<UNIT
[Unit]
Description=T-Killer live loop
[Service]
Type=oneshot
User=$USER
ExecStart=$DIR/deploy/run.sh live
TimeoutStartSec=240
UNIT
sudo tee /etc/systemd/system/tk-live.timer >/dev/null <<UNIT
[Unit]
Description=T-Killer live loop every minute
[Timer]
OnCalendar=*-*-* *:*:20
AccuracySec=5s
Persistent=false
[Install]
WantedBy=timers.target
UNIT
sudo tee /etc/systemd/system/tk-daily.service >/dev/null <<UNIT
[Unit]
Description=T-Killer nightly research and plan
[Service]
Type=oneshot
User=$USER
ExecStart=$DIR/deploy/run.sh daily
TimeoutStartSec=3600
UNIT
sudo tee /etc/systemd/system/tk-daily.timer >/dev/null <<UNIT
[Unit]
Description=T-Killer nightly run at 22:45 UTC
[Timer]
OnCalendar=*-*-* 22:45:00
Persistent=true
[Install]
WantedBy=timers.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now tk-live.timer tk-daily.timer
systemctl list-timers 'tk-*' --no-pager

echo
echo "Done. Mode is PAPER (no orders). Logs: journalctl -u tk-live -n 50"
echo "Switch to testnet later with: sed -i \"s/^MODE=.*/MODE='exchange'/\" $DIR/.env"
