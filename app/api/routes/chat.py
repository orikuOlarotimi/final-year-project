from fastapi import APIRouter, Depends, HTTPException, Query
from app.models.chat import Chat
from app.core.dependencies import get_current_user
from app.models.message import Message
from app.models.document import DocumentModel
from app.schemas.schemas import ChatHistoryResponse
from app.services.memory_service import build_conversation_pairs

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


@router.get("/{chat_id}/messages", response_model=ChatHistoryResponse)
async def get_chat_history(
    chat_id: str,
    document_id: str,
    limit: int = Query(10, ge=1, le=50),
    user_id: str = Depends(get_current_user)

):
    try:
        chat_id = chat_id.strip()
        document_id = document_id.strip()

        if not chat_id or not document_id:
            raise HTTPException(
                status_code=400,
                detail={"success": False, "message": "chat_id and document_id are required"})

        chat = await Chat.get(chat_id)
        if not chat or str(chat.user_id) != str(user_id):
            raise HTTPException(
                status_code=404,
                detail={"success": False, "message": "Chat not found"}
            )
        doc = await DocumentModel.get(document_id)
        if not doc or str(doc.chat_id) != str(chat_id):
            raise HTTPException(
                status_code=404,
                detail={"success": False, "message": "Document not found in this chat"}
            )
        # 🔹 1. Fetch messages (latest first)
        messages = await Message.find(
            Message.chat_id == chat_id,
            Message.user_id == user_id,
            Message.document_id == document_id
        ).sort("-created_at").limit(limit).to_list()

        if not messages:
            return {
                "success": True,
                "messages": [],
                "pairs": []
            }

        # 🔹 2. Reverse → oldest first (important for chat order)
        messages.reverse()

        # 🔹 3. Format response
        formatted = [
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat()
            }
            for msg in messages
        ]
        simple_messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
        print(simple_messages)
        pairs = build_conversation_pairs(simple_messages)

        return {
            "success": True,
            "messages": formatted,  # UI
            "pairs": pairs  # 🔥 memory-ready
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Failed to fetch chat history",
                "error": str(e)
            }
        )
