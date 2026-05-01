from langchain.tools import tool
from app.core.vector_store import vector_store


def create_retrieval_tool(user_id: str, chat_id: str, document_id: str | None = None):
    @tool
    def retrieve_documents(query: str) -> str:
        """
        Retrieve relevant document chunks for answering user questions.
        Use this when the user asks anything related to their uploaded documents.
        """
        docs = vector_store.similarity_search(
            query,
            k=5,
            namespace=user_id,  # ✅ captured from outer scope
            filter={"chat_id": chat_id, "document_id": document_id}  # ✅ captured from outer scope
        )

        if not docs:
            return "No relevant information found in the uploaded documents."

        return "\n\n".join([doc.page_content for doc in docs])

    return retrieve_documents
