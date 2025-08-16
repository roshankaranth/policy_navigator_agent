from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredXMLLoader
from dotenv import load_dotenv
import os
import glob
import asyncio
import time
import uuid
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

client = QdrantClient(url="http://localhost:6333")
client.recreate_collection(
    collection_name="Policy_Navigator_dataset",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=20))
def safe_embed(texts):
    return embedding_model.embed_documents(texts)

def enrich_metadata(doc: Document, filename: str) -> Document:
    """
    Add filename and possibly a section title to metadata.
    """
    metadata = doc.metadata or {}
    metadata["filename"] = os.path.basename(filename)

    # Optional: Extract heading/title from text
    lines = doc.page_content.strip().splitlines()
    for line in lines:
        if line.strip() and len(line.strip()) < 100:
            metadata["section_title"] = line.strip()
            break

    doc.metadata = metadata
    return doc

async def ingest_xml_files():
    dataset_dir = "../dataset/CodeOfRegulation"
    xml_files = glob.glob(os.path.join(dataset_dir, "*.xml"))
    pages = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    skipped_batches = []

    total_docs = len(xml_files)
    for idx, file_path in enumerate(xml_files, 1):
        print(f"Loading file: {file_path}, progress: {idx}/{total_docs}")
        loader = UnstructuredXMLLoader(file_path)
        async for page in loader.alazy_load():
            enriched = enrich_metadata(page, file_path)
            pages.append(enriched)

    print(f"Splitting documents... Number of pages: {len(pages)}")
    all_splits = text_splitter.split_documents(pages)
    print(f"Number of splits: {len(all_splits)}")

    # Batch embed and upsert
    batch_size = 20
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        try:
            embeddings = safe_embed(texts)
        except Exception as e:
            print(f"Failed to embed batch {i}-{i + batch_size}: {e}")
            skipped_batches.append({
                "start": i,
                "end": i + batch_size,
                "error": str(e),
                "filenames": list({meta.get("filename", "unknown") for meta in metadatas})
            })
            continue

        points = []
        for emb, meta in zip(embeddings, metadatas):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload=meta
            ))

        client.upsert(collection_name="Policy_Navigator_dataset", points=points)
        print(f"Upserted batch {i}-{i + batch_size}")

    print("Ingestion complete!")

    # Log skipped batches
    if skipped_batches:
        with open("skipped_batches.log", "w") as f:
            for batch in skipped_batches:
                f.write(f"{batch}\n")
        print(f"Logged {len(skipped_batches)} skipped batches to skipped_batches.log")

async def ingestion_pipeline():
    await ingest_xml_files()

async def main():
    start = time.time()
    await ingestion_pipeline()
    end = time.time()
    print(f"Time elapsed: {end - start:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
