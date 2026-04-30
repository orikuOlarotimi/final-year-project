import requests
from app.core.config import BREVO_API_KEY
from app.core.config import EMAIL_FROM
class EmailService:

    def send_otp_email(self, email: str, otp: str):
        url = "https://api.brevo.com/v3/smtp/email"

        headers = {
            "accept": "application/json",
            "api-key": BREVO_API_KEY,
            "content-type": "application/json"
        }

        payload = {
            "sender": {
                "name": "Your App",
                "email": EMAIL_FROM
            },
            "to": [{"email": email}],
            "subject": "Your OTP Code",
            "htmlContent": f"<h2>Your OTP is: {otp}</h2>"
        }

        response = requests.post(url, json=payload, headers=headers)

        return response.json()