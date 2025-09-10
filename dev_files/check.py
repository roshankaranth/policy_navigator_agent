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

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
cosmos_client = CosmosClient(cosmos_host, cosmos_key)
database_name = "vectordb"
container_name = "embeddings"
partition_key = PartitionKey(path="/userId")
cosmos_container_properties = {"partition_key": partition_key}

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

# vector_search_fields = {
#     "embedding_field": "embedding", 
#     "text_field": "text",         
# }

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
    # vector_search_fields=vector_search_fields,
)

def retrival_pipeline(query : str):
    print("Retrieving context..")
    retrieved_docs = vector_search.similarity_search(query=query, k=5)
    #
    # print(retrieved_docs)
    return retrieved_docs
    
# database = cosmos_client.get_database_client(database_name)
# container = database.get_container_client(container_name)

# query = "SELECT TOP 1 * FROM c ORDER BY c._ts DESC"
# items = list(container.query_items(
#     query=query,
#     enable_cross_partition_query=True
# ))

# if items:
#     last_document = items[0]
#     print("Last updated document found:")
#     print(last_document["metadata"]["source"])
# else:
#     print("Container is empty or no documents were found.")

# timestamp = last_document['_ts']

# # Convert it to a readable UTC datetime object
# utc_datetime = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)

# # Convert it to your local timezone (e.g., IST)
# ist_timezone = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
# local_time = utc_datetime.astimezone(ist_timezone)

retrival_pipeline("Hello")