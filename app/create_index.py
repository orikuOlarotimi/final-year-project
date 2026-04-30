import os
from pinecone import Pinecone
from dotenv import load_dotenv
from pinecone import ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

pc.create_index(
    name="rag-project-main",
    dimension=3072,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
