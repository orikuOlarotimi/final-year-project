from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from app.models.message import Message


def build_conversation_pairs(messages):
    """
    Convert raw messages into safe (input, output) pairs
    """
    pairs = []
    current_user_message = None

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            # always take the latest user intent
            current_user_message = content

        elif role == "assistant":
            if current_user_message:
                pairs.append({
                    "input": current_user_message,
                    "output": content
                })
                current_user_message = None
            # else → ignore orphan assistant message
    return pairs


def build_memory_from_pairs(pairs):
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,   # adjust later
        return_messages=True
    )

    for pair in pairs:
        memory.save_context(
            {"input": pair["input"]},
            {"output": pair["output"]}
        )

    return memory

async def load_memory(chat_id: str, user_id: str, document_id: str):
    messages = await Message.find(
        Message.chat_id == chat_id,
        Message.user_id == user_id,
        Message.document_id == document_id
    ).sort("created_at").limit(100).to_list()

    if not messages:
        return None

    # format like your endpoint
    formatted = [
        {
            "role": msg.role,
            "content": msg.content
        }
        for msg in messages
    ]

    pairs = build_conversation_pairs(formatted)

    if not pairs:
        return None

    memory = build_memory_from_pairs(pairs)

    return memory
