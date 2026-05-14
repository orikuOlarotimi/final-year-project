from beanie import Document
from pydantic import EmailStr, Field
from datetime import datetime


class User(Document):
    email: EmailStr = Field(..., unique=True)
    password: str  # hashed password
    name: str = Field(..., min_length=2, max_length=100)
    is_active: bool = True
    is_verified: bool = False

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "users"
