from app.services.in_memory_store import chat_memory_store


class ChatMemoryService:

    @staticmethod
    def _get_key(user_id: str, chat_id: str, document_id: str):
        return f"{user_id}:{chat_id}:{document_id}"

    @staticmethod
    def get_memory(user_id: str, chat_id: str, document_id: str):
        key = ChatMemoryService._get_key(user_id, chat_id, document_id)

        if key not in chat_memory_store:
            chat_memory_store[key] = {
                "messages": []
            }

        return chat_memory_store[key]

    @staticmethod
    def set_memory(user_id: str, chat_id: str, document_id: str, messages: list):
        key = ChatMemoryService._get_key(user_id, chat_id, document_id)

        chat_memory_store[key] = {
            "messages": messages
        }

    @staticmethod
    def add_message(user_id: str, chat_id: str, document_id: str, role: str, content: str):
        memory = ChatMemoryService.get_memory(user_id, chat_id, document_id)

        memory["messages"].append({
            "role": role,
            "content": content
        })

        # keep last 20 messages
        if len(memory["messages"]) > 20:
            memory["messages"] = memory["messages"][-20:]

    @staticmethod
    def clear_memory(user_id: str, chat_id: str, document_id: str):
        key = ChatMemoryService._get_key(user_id, chat_id, document_id)

        if key in chat_memory_store:
            del chat_memory_store[key]