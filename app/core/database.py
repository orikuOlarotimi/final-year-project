from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import os
from pymongo import AsyncMongoClient
# import all models
from app.models.user import User
from app.models.chat import Chat
from app.models.message import Message
from app.models.document import DocumentModel
from app.models.token import RefreshToken
from app.models.otp import OTP


from dotenv import load_dotenv
load_dotenv()


async def init_db():
    client = AsyncMongoClient(os.getenv("MONGO_URI"))

    db = client["rag_project_db"]

    await init_beanie(
        database=db,
        document_models=[
            User,
            Chat,
            Message,
            DocumentModel,
            OTP,
            RefreshToken
        ]
    )
