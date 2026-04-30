import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "rag-project-main"

# Connect to index
index = pc.Index(index_name)

# Initialize embeddings (handled by LangChain)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# Create vector store (THIS is the key part)
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)
