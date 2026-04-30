from beanie import Document
from pydantic import Field
from datetime import datetime
from typing import Optional


class Chat(Document):
    user_id: str = Field(..., index=True)

    title: Optional[str] = "New Chat"

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "chats"