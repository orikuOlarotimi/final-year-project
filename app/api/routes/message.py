from fastapi import APIRouter, FastAPI, HTTPException, Depends
from app.models.message import Message
from app.core.dependencies import get_current_user
from app.services.agent_service import run_agent
from app.schemas.schemas import ChatQuerySchema
from app.models.chat import Chat
from app.models.document import DocumentModel


router = APIRouter(prefix="/message", tags=["Messages"])


@router.post("/message")
async def send_message(payload: ChatQuerySchema, user_id: str = Depends(get_current_user)):

    # 🔹 validate chat ownership
    chat = await Chat.get(payload.chat_id)
    if not chat or str(chat.user_id) != str(user_id):
        raise HTTPException(status_code=404, detail={"success": False, "message": "Chat not found"})

    if payload.document_id:
        doc = await DocumentModel.get(payload.document_id)
        if not doc or str(doc.chat_id) != str(payload.chat_id):
            raise HTTPException(status_code=404, detail={"success": False, "message": "Document not found in this chat"})

    # 🔹 1. Save USER message
    user_msg = Message(
        chat_id=payload.chat_id,
        user_id=user_id,
        role="user",
        content=payload.message.strip()
    )
    await user_msg.insert()

    # 🔹 2. Run agent
    response = await run_agent(
        user_id=user_id,
        chat_id=payload.chat_id,
        question=payload.message,
        document_id=payload.document_id
    )

    # 🔹 3. Save ASSISTANT message
    bot_msg = Message(
        chat_id=payload.chat_id,
        user_id=user_id,
        role="assistant",
        content=response["answer"]
    )
    await bot_msg.insert()

    return response
