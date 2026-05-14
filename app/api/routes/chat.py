from fastapi import APIRouter, Depends, HTTPException, Query
from app.models.chat import Chat
from app.core.dependencies import get_current_user
from app.models.message import Message
from app.models.document import DocumentModel
from app.schemas.schemas import ChatHistoryResponse
from app.services.memory_service import build_conversation_pairs
from app.services.chat_memory_service import ChatMemoryService
from beanie import PydanticObjectId
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
async def get_user_chats(
        user_id: str = Depends(get_current_user),
):
    try:

        chats = (
            await Chat.find(Chat.user_id == user_id)
            .sort("-updated_at")
            .to_list()
        )

        # Defensive check for empty chats
        if not chats:
            return {
                "success": True,
                "count": 0,
                "message": "No chats found",
                "chats": [],
            }

        formatted_chats = []

        for chat in chats:

            try:
                # Fetch first message for preview
                first_message = await (
                    Message.find(
                        Message.chat_id == str(chat.id),
                        Message.user_id == user_id,
                    )
                    .sort("created_at")
                    .first_or_none()
                )

                formatted_chats.append(
                    {
                        "chat_id": str(chat.id),

                        "preview": (
                            first_message.content[:60]
                            if first_message
                               and first_message.content
                            else "New Chat"
                        ),

                        "created_at": chat.created_at,
                        "updated_at": chat.updated_at,
                    }
                )

            except Exception as message_error:
                print(
                    f"Error processing chat "
                    f"{str(chat.id)}: {message_error}"
                )

                # Continue processing other chats
                formatted_chats.append(
                    {
                        "chat_id": str(chat.id),
                        "preview": "Unable to load preview",
                        "created_at": chat.created_at,
                        "updated_at": chat.updated_at,
                    }
                )

        return {
            "success": True,
            "count": len(formatted_chats),
            "message": "Chats fetched successfully",
            "chats": formatted_chats,
        }

    except Exception as error:
        print(f"Get chats error: {error}")

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Failed to fetch chats",
            },
        )


@router.get("/{chat_id}")
async def get_chat(
        chat_id: str,
        user_id: str = Depends(get_current_user),
):
    try:
        print(chat_id)
        # Ensure chat belongs to current user
        chat = await Chat.find_one(
            Chat.id == PydanticObjectId(chat_id),
            Chat.user_id == user_id,
        )

        if not chat:
            raise HTTPException(
                status_code=404,
                detail={"success": False, "message": "Chat not found"},
            )

        # Fetch all messages in ascending order
        messages = (
            await Message.find(
                Message.chat_id == chat_id,
                Message.user_id == user_id,
            )
            .sort("created_at")
            .to_list()
        )

        # Fetch all uploaded documents for this chat
        documents = (
            await DocumentModel.find(
                DocumentModel.chat_id == chat_id,
                DocumentModel.user_id == user_id,
            )
            .sort("-created_at")
            .to_list()
        )

        formatted_messages = []

        for message in messages:
            formatted_messages.append({
                "message_id": str(message.id),
                "role": message.role,
                "content": message.content,
                "document_id": message.document_id,
                "created_at": message.created_at,
            })

        formatted_documents = []

        for document in documents:
            formatted_documents.append({
                "document_id": str(document.id),
                "filename": document.filename,
                "file_type": document.file_type,
                "created_at": document.created_at,
            })

        return {
            "success": True,

            "chat": {
                "chat_id": str(chat.id),
                "title": chat.title,
                "active_document_id": chat.active_document_id,
                "created_at": chat.created_at,
                "updated_at": chat.updated_at,
            },

            "documents": formatted_documents,

            "messages": formatted_messages,
        }
    except HTTPException:
        raise
    except Exception as error:
        print(f"Get chat error: {error}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Failed to fetch chat",
            },
        )


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
            ChatMemoryService.set_memory(user_id, chat_id, document_id, [])
            return {
                "success": True
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
        ChatMemoryService.set_memory(user_id, chat_id, document_id, simple_messages)

        return {
            "success": True,
            "messages": formatted
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
