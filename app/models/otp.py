from beanie import Document
from datetime import datetime, timedelta
from pydantic import EmailStr, Field


class OTP(Document):
    email: EmailStr
    code: str

    expires_at: datetime
    # is_used: bool = False

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "otps"