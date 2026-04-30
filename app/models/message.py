from beanie import Document
from pydantic import Field
from datetime import datetime


class Message(Document):
    chat_id: str = Field(..., index=True)
    user_id: str = Field(..., index=True)

    role: str  # "user" or "assistant"
    content: str

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "messages"