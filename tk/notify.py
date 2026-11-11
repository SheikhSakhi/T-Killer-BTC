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


def brevo(subject, body, sender, to):
    """Email over HTTPS (api.brevo.com), for servers whose outbound SMTP is cut. Free plan: 300 emails a day.
    The sender address must be verified once in Brevo (Senders, Domains & Dedicated IPs → Senders)."""
    r = requests.post("https://api.brevo.com/v3/smtp/email", timeout=30,
                      headers={"api-key": os.environ["BREVO_API_KEY"], "accept": "application/json"},
                      json=dict(sender=dict(email=sender), to=[dict(email=to)], subject=subject, textContent=body))
    if not r.ok:
        raise OSError(f"brevo {r.status_code}: {r.text[:200]}")
    return True


def email(subject, body):
    user, pwd = os.environ.get("GMAIL_USER"), os.environ.get("GMAIL_APP_PASSWORD")
    to = os.environ.get("ALERT_EMAIL_TO") or user
    if os.environ.get("BREVO_API_KEY") and user:
        return brevo(subject, body, user, to)
    if not (user and pwd):
        return False
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    errors = []
    for port in (465, 587):         # 465 = SSL from the start; 587 = plain then STARTTLS (some hosts cut one of them)
        try:
            with (smtplib.SMTP_SSL if port == 465 else smtplib.SMTP)("smtp.gmail.com", port, timeout=30) as s:
                if port == 587:
                    s.starttls()
                s.login(user, pwd)
                s.sendmail(user, [to], msg.as_string())
            return True
        except smtplib.SMTPAuthenticationError:
            raise                   # wrong address or app password: another port will not help
        except Exception as e:
            errors.append(f"{port}: {e!r}")
    raise OSError("gmail unreachable on both ports; " + "; ".join(errors))


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
