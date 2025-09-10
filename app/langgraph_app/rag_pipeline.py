from langchain_google_genai import GoogleGenerativeAIEmbeddings
from azure.cosmos import CosmosClient, PartitionKey
#from langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql import AzureCosmosDBNoSqlVectorSearch
from langchain_community.vectorstores.azure_cosmos_db_no_sql import (
    AzureCosmosDBNoSqlVectorSearch,
)
import os
from dotenv import load_dotenv

load_dotenv()

cosmos_host = os.getenv("COSMOS_HOST")
cosmos_key = os.getenv("COSMOS_KEY")

cosmos_client = CosmosClient(cosmos_host, cosmos_key)
database_name = "vectordb"
container_name = "embeddings"
partition_key = PartitionKey(path="/userId")
cosmos_container_properties = {"partition_key": partition_key}
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")


indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
    "fullTextIndexes": [{"path": "/text"}],
}

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

full_text_policy = {
    "defaultLanguage": "en-US",
    "fullTextPaths": [{"path": "/text", "language": "en-US"}],
}


vector_search = AzureCosmosDBNoSqlVectorSearch(
    embedding=embeddings,
    cosmos_client=cosmos_client,
    vector_embedding_policy = vector_embedding_policy,
    indexing_policy = indexing_policy,
    database_name=database_name,
    container_name=container_name,
    full_text_policy=full_text_policy,
    cosmos_container_properties=cosmos_container_properties,
    cosmos_database_properties={},
    full_text_search_enabled=True,
)   

def retrival_pipeline(query : str):
    
    retrieved_docs = vector_search.similarity_search(query=query, k=5)
    return retrieved_docs
    
