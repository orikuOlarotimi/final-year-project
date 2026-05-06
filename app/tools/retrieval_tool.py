from langchain.tools import tool
from app.core.vector_store import vector_store
from app.services.chat_memory_service import ChatMemoryService
from langchain_openai import ChatOpenAI


def create_retrieval_tool(user_id: str, chat_id: str, document_id: str | None = None):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    @tool
    def retrieve_documents(query: str) -> str:
        """
        Retrieve relevant document chunks for answering user questions.
        Use this when the user asks anything related to their uploaded documents.
        """
        if not document_id:
            return (
                "NO_DOCUMENT_SELECTED: The user has not selected a document. "
                "Ask the user to choose a document before retrieving information."
            )
        memory = ChatMemoryService.get_memory(user_id, chat_id, document_id)
        history = memory.get("messages", [])[-4:]  # last 4–8 turns

        formatted_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in history
        )
        QUERY_REWRITE_PROMPT = """
                You are a query rewriting system for document retrieval.

                Task:
                Rewrite the user's question into a clear, keyword-rich search query
                for a vector database.

                Rules:
                - Do NOT answer the question
                - Do NOT add new information
                - Only extract and clarify intent
                - Output ONLY the rewritten query, nothing else — no explanation, no preamble
                - If chat history is empty, return the original question exactly as written

                User question:
                {question}

                Chat history:
                {history}
                Rewritten query:
                """
        rewritten_query = llm.invoke(
            QUERY_REWRITE_PROMPT.format(
                question=query,
                history=formatted_history
            )
        ).content.strip()

        docs = vector_store.similarity_search(
            rewritten_query,
            k=5,
            namespace=user_id,  # ✅ captured from outer scope
            filter={"chat_id": chat_id, "document_id": document_id}  # ✅ captured from outer scope
        )

        if not docs:
            return "No relevant information found in the uploaded documents."

        return "\n\n".join([doc.page_content for doc in docs])

    return retrieve_documents
