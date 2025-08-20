from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient 
import os
import glob
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

client = QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="test_docs",
    embedding=embeddings,
)

def retrival_pipeline(query : str):
    print("Retrieving context..")
    retrieved_docs = vector_store.similarity_search(query)
    return retrieved_docs
    