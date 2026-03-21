"""Email service using Resend for transactional email delivery."""

import logging
import os

import resend

logger = logging.getLogger(__name__)

resend.api_key = os.getenv("RESEND_API_KEY", "")

_FROM_ADDRESS = "reelAI <onboarding@resend.dev>"


def send_welcome_email(*, email: str, username: str) -> None:
    """Send a welcome email to a newly registered user.

    Errors are caught and logged so that a failure here never blocks
    the registration response from reaching the client.
    """
    api_key = os.getenv("RESEND_API_KEY", "").strip()
    if not api_key:
        logger.warning("RESEND_API_KEY is not configured — skipping welcome email for %s", email)
        return

    greeting = f"@{username}" if str(username or "").strip() else "there"
    html_body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Welcome to reelAI</title>
</head>
<body style="margin:0;padding:0;background-color:#0f0f0f;font-family:Arial,Helvetica,sans-serif;color:#f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0f0f0f;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color:#1a1a1a;border-radius:8px;overflow:hidden;max-width:600px;width:100%;">
          <tr>
            <td style="padding:40px 48px 32px;">
              <h1 style="margin:0 0 8px;font-size:28px;font-weight:700;color:#ffffff;">Welcome to reelAI</h1>
              <p style="margin:0 0 24px;font-size:16px;color:#a0a0a0;">Your account is ready.</p>
              <p style="margin:0 0 16px;font-size:16px;line-height:1.6;color:#e0e0e0;">
                Hi {greeting},
              </p>
              <p style="margin:0 0 16px;font-size:16px;line-height:1.6;color:#e0e0e0;">
                Thanks for joining reelAI! Your account has been created and you're all set to start exploring.
              </p>
              <p style="margin:0 0 32px;font-size:16px;line-height:1.6;color:#e0e0e0;">
                Dive in and start creating — we're excited to have you on board.
              </p>
              <p style="margin:0;font-size:14px;color:#606060;">
                If you didn't create this account, you can safely ignore this email.
              </p>
            </td>
          </tr>
          <tr>
            <td style="padding:24px 48px;border-top:1px solid #2a2a2a;">
              <p style="margin:0;font-size:13px;color:#606060;">
                &copy; reelAI. All rights reserved.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

    try:
        resend.api_key = api_key
        resend.Emails.send({
            "from": _FROM_ADDRESS,
            "to": [email],
            "subject": "Welcome to reelAI!",
            "html": html_body,
        })
        logger.info("Welcome email sent to %s", email)
    except Exception:
        logger.exception("Failed to send welcome email to %s", email)
