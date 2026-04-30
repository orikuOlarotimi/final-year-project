from beanie import Document
from pydantic import Field
from datetime import datetime
from typing import Optional


class DocumentModel(Document):
    user_id: str = Field(..., index=True)
    chat_id: str = Field(..., index=True)

    filename: str
    file_path: str

    file_type: Optional[str] = None
    file_size: Optional[int] = None

    status: str = Field(default="uploaded")
    # uploaded → processing → processed → failed

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "documents"
