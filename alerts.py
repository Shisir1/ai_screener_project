from __future__ import annotations
import os
import smtplib
from email.mime.text import MIMEText
from typing import Optional

def send_email(subject: str, body: str, to_email: str, from_email: Optional[str] = None):
    """ Uses SMTP settings from env:
    SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
    for Gmail you likely need to use an App Password.
    """

    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASS")

    if not host or not user or not pwd:
        raise RuntimeError("Missing SMTP env vars. Fill SMTP_HOST, SMTP_USER, SMTP_PASS.")
    
    from_email = from_email or user
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, pwd)
        server.sendmail(from_email, [to_email], msg.as_string())