import smtplib

import pytest

from tk import notify


class FakeSMTP:
    sent = []

    def __init__(self, host, port, timeout=None):
        if port in self.broken:
            raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")
        self.port = port

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self):
        pass

    def login(self, u, p):
        pass

    def sendmail(self, f, to, m):
        FakeSMTP.sent.append(self.port)


def setup(monkeypatch, broken):
    FakeSMTP.broken, FakeSMTP.sent = broken, []
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("GMAIL_USER", "bot@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "x")


def test_falls_back_to_587_when_465_is_cut(monkeypatch):
    setup(monkeypatch, {465})
    assert notify.email("s", "b") is True and FakeSMTP.sent == [587]


def test_reports_both_ports_when_gmail_unreachable(monkeypatch):
    setup(monkeypatch, {465, 587})
    with pytest.raises(OSError, match="465.*587"):
        notify.email("s", "b")
