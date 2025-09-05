import os
import time
from langchain_community.document_loaders import DirectoryLoader, UnstructuredXMLLoader
from azure.cosmos import CosmosClient, PartitionKey
from langchain_community.vectorstores.azure_cosmos_db_no_sql import (
    AzureCosmosDBNoSqlVectorSearch,
)
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm

load_dotenv()

cosmos_host = os.getenv("COSMOS_HOST")
cosmos_key = os.getenv("COSMOS_KEY")
if not cosmos_host or not cosmos_key:
    raise ValueError("COSMOS_HOST and COSMOS_KEY must be set in your .env file.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
cosmos_client = CosmosClient(cosmos_host, cosmos_key)

database_name = "vectordb"
container_name = "embeddings"

cosmos_container_properties = {"partition_key": PartitionKey(path="/userId")}

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 768,
        }
    ]
}

indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
}

print("Loading and splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
loader = DirectoryLoader(
    path="test_dataset",
    glob="**/*.xml",
    loader_cls=UnstructuredXMLLoader,
)
docs = loader.load()
print(f"Total documents found: {len(docs)}")

all_splits = text_splitter.split_documents(docs)

for split in all_splits:
    if "userId" not in split.metadata:
        split.metadata["userId"] = "default_user_1" # Assign a default partition key value

print(f"All documents processed. Total chunks: {len(all_splits)}")

print(" Initializing Azure Cosmos DB Vector Store...")
vector_store = AzureCosmosDBNoSqlVectorSearch(
    cosmos_client=cosmos_client,
    embedding=embeddings,
    database_name=database_name,
    container_name=container_name,
    vector_embedding_policy=vector_embedding_policy,
    indexing_policy=indexing_policy,
    cosmos_container_properties=cosmos_container_properties,
    cosmos_database_properties={},
)

print("\n Uploading chunks to Azure Cosmos DB in batches...")
batch_size = 100
failed_documents_to_retry = []

for i in tqdm(range(0, len(all_splits), batch_size), desc="Uploading Batches"):
    batch = all_splits[i:i + batch_size]

    try:
        vector_store.add_documents(batch)

    except Exception as e:
        batch_number = i // batch_size + 1
        print(f"\n Error processing batch {batch_number}: {e}")

        failed_documents_to_retry.extend(batch)

        time.sleep(2)

print("\n All chunks processed.")

if failed_documents_to_retry:
    print(f"\n Warning: {len(failed_documents_to_retry)} document chunks failed to upload.")

    failed_files = set(doc.metadata.get('source', 'Unknown Source') for doc in failed_documents_to_retry)

    output_filename = "failed_files.txt"
    try:
        with open(output_filename, "w") as f:
            for filename in sorted(list(failed_files)):
                f.write(f"{filename}\n")
        print(f"\n📄 A list of files that had upload errors has been saved to: {output_filename}")
    except IOError as e:
        print(f"\n Could not write to file {output_filename}: {e}")
else:
    print("\n All documents were ingested successfully!")