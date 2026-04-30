from fastapi import APIRouter, Depends, HTTPException
from app.models.chat import Chat
from app.core.dependencies import get_current_user

router = APIRouter(prefix="/chats", tags=["Chats"])


@router.post("/")
async def create_chat(user_id: str = Depends(get_current_user)):
    try:
        chat = Chat(
            user_id=user_id,
            title="New Chat"
        )

        await chat.insert()

        return {
            "success": True,
            "chat_id": str(chat.id),
            "message": "Chat created"
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "Failed to create chat"}
        )


@router.get("/")
async def get_user_chats(user_id: str = Depends(get_current_user)):
    chats = await Chat.find(Chat.user_id == user_id).sort("-updated_at").to_list()

    return {
        "success": True,
        "chats": chats
    }

@router.get("/{chat_id}")
async def get_chat(chat_id: str, user_id: str = Depends(get_current_user)):
    chat = await Chat.find_one(
        Chat.id == chat_id,
        Chat.user_id == user_id
    )

    if not chat:
        raise HTTPException(
            status_code=404,
            detail={"success": False, "message": "Chat not found"}
        )

    return {
        "success": True,
        "chat": chat
    }