"""Alerts to Telegram and email. Missing credentials just skip that channel."""
import os
import smtplib
import sys
from email.mime.text import MIMEText

import requests


def telegram(text):
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data=dict(chat_id=chat, text=text[:4000], disable_web_page_preview="true"), timeout=20)
    return r.ok


def email(subject, body):
    user, pwd = os.environ.get("GMAIL_USER"), os.environ.get("GMAIL_APP_PASSWORD")
    to = os.environ.get("ALERT_EMAIL_TO") or user
    if not (user and pwd):
        return False
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
        s.login(user, pwd)
        s.sendmail(user, [to], msg.as_string())
    return True


def send(subject, body, mail=True):
    """Telegram always; email for things worth keeping (entries, exits, daily report, errors)."""
    ok = True
    for fn, args in ((telegram, (f"{subject}\n\n{body}",)), (email, (subject, body)) if mail else (None, None)):
        if fn is None:
            continue
        try:
            fn(*args)
        except Exception as e:  # an alert failure must never stop trading logic
            ok = False
            print(f"notify {fn.__name__} failed: {e}", file=sys.stderr)
    return ok


def chat_id_hint():
    """Print chat ids that messaged the bot, to fill TELEGRAM_CHAT_ID during setup."""
    token = os.environ.get("TELEGRAM_TOKEN")
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20).json()
    ids = {u["message"]["chat"]["id"] for u in r.get("result", []) if "message" in u}
    print("\n".join(map(str, ids)) or "No messages yet: send /start to your bot, then run this again.")


if __name__ == "__main__":
    if sys.argv[1:] == ["chat-id"]:
        chat_id_hint()
    else:
        print(send("T-Killer test", "If you can read this, alerts work."))
