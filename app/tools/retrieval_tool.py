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

        contexts = [doc.page_content for doc in docs]
        print("answerssss..", contexts, "the END")

        if not docs:
            return "No relevant information found in the uploaded documents."

        def _format_doc(doc) -> str:
            meta = doc.metadata
            source = meta.get("source", "")
            filename = source.split("/")[-1] if source else meta.get("document_id", "Unknown file")
            page = meta.get("page")
            modality = meta.get("modality", "text")
            file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

            # Human-readable type label so the LLM can immediately see what
            # kind of content this chunk is, without having to infer it from
            # the page_content text alone.
            type_labels = {
                "text":  "Text",
                "ocr":   "Text (OCR)",
                "image": "Figure",
                "table": "Table",
            }
            type_label = type_labels.get(modality, modality.capitalize())

            if file_type == "pdf" and page is not None:
                label = f"[Source: {filename}, Page {page + 1}, Type: {type_label}]"
            else:
                label = f"[Source: {filename}, Type: {type_label}]"

            return f"{label}\n{doc.page_content}"

        return "\n\n".join([_format_doc(doc) for doc in docs])

    return retrieve_documents