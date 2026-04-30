from app.core.vector_store import vector_store
from langchain_core.documents import Document

docs = [
    Document(page_content="LangChain makes RAG easier"),
]

vector_store.add_documents(docs)