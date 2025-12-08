"""Email service implementation using SMTP."""

import os
from email.message import EmailMessage

import aiosmtplib
import dotenv

dotenv.load_dotenv()

SENDER = os.getenv("EMAIL_SENDER", "Eatsential <noreply@eatsential.com>")
SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("SMTP_PORT", "1025"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

VERIFICATION_TEMPLATE = """
Welcome to Eatsential!

Please verify your email address by clicking the link below:

{verification_url}

This link will expire in 24 hours.

Best regards,
The Eatsential Team
"""

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")


async def send_verification_email(email: str, token: str) -> bool:
    """Send verification email using SMTP

    Args:
        email: Recipient email address
        token: Verification token

    Returns:
        True if email was sent successfully

    Raises:
        SMTPException: If email sending fails

    """
    verification_url = f"{FRONTEND_URL}/verify-email?token={token}"

    message = EmailMessage()
    message["From"] = SENDER
    message["To"] = email
    message["Subject"] = "Please verify your Eatsential email address"
    message.set_content(VERIFICATION_TEMPLATE.format(verification_url=verification_url))

    try:
        await aiosmtplib.send(
            message,
            hostname=SMTP_HOST,
            port=SMTP_PORT,
            username=SMTP_USER or None,
            password=SMTP_PASSWORD or None,
            use_tls=bool(SMTP_USER and SMTP_PASSWORD),
        )
        return True
    except Exception as e:
        print(f"Failed to send email: {e!s}")
        return False
        ##print(f"📧 Verification email for {email}: {verification_url}")
        ##return True  # Return True anyway so user can proceed in development
