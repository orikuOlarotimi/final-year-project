from beanie import Document
from datetime import datetime
from pydantic import Field


class RefreshToken(Document):
    user_id: str
    token_hash: str

    expires_at: datetime

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "refresh_tokens"