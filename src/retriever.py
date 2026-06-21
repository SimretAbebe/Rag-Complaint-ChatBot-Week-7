import os
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vector_store",
    "prebuilt_store"
)
COLLECTION_NAME = "complaints_full"

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded!")


def get_collection():
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    return collection


def retrieve(question: str,
             collection,
             n_results: int = 5,
             product_filter: str = None) -> list:
   

   
    question_embedding = embedding_model.encode(question).tolist()

   
    where_filter = None
    if product_filter:
        where_filter = {"product_category": product_filter}

    
    if where_filter:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
    else:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

   
    retrieved_chunks = []
    for i in range(len(results["documents"][0])):
        chunk = {
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity_score": round(1 - results["distances"][0][i], 4)
        }
        retrieved_chunks.append(chunk)

    return retrieved_chunks


def format_context(retrieved_chunks: list) -> str:
    
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        part = (
            f"[Complaint {i}]\n"
            f"Product: {meta.get('product_category', 'Unknown')}\n"
            f"Issue: {meta.get('issue', 'Unknown')}\n"
            f"Company: {meta.get('company', 'Unknown')}\n"
            f"Text: {chunk['text']}\n"
        )
        context_parts.append(part)

    return "\n---\n".join(context_parts)