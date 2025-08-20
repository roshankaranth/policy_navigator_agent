import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredXMLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Qdrant config
QDRANT_URL = "http://localhost:6333"   # or Qdrant Cloud URL
COLLECTION_NAME = "test_docs"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

loader = DirectoryLoader(
    path="../test_dataset", 
    glob="**/*.xml",  # recursively find all XML files
    loader_cls=UnstructuredXMLLoader,
)

docs = loader.load()
print(f"No. of docs : len(docs)")
all_splits = text_splitter.split_documents(docs)
print(f"Loaded {len(all_splits)} XML chunks")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = QdrantVectorStore.from_documents(
    documents=all_splits,
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME
)

print("All XML docs ingested into Qdrant with Gemini embeddings")
