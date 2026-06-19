from __future__ import annotations

import argparse
import math
import os
from typing import List

import pandas as pd

from src.preprocessing import load_filtered_dataset, ensure_columns
from src.chunking import chunk_with_metadata


def stratified_sample(df: pd.DataFrame, target_n: int = 12000, key: str = "product_category") -> pd.DataFrame:
    if df.empty:
        raise ValueError("Input dataframe is empty")

    groups = df.groupby(key)
    counts = groups.size().to_dict()
    total = len(df)

    # initial allocation proportionally
    alloc = {g: max(1, int(round(target_n * (count / total)))) for g, count in counts.items()}

    # adjust to match target_n
    current = sum(alloc.values())
    sorted_groups = sorted(counts.items(), key=lambda x: -x[1])
    i = 0
    while current != target_n:
        grp = sorted_groups[i % len(sorted_groups)][0]
        if current < target_n:
            alloc[grp] += 1
            current += 1
        else:
            if alloc[grp] > 1:
                alloc[grp] -= 1
                current -= 1
        i += 1

    parts = []
    for grp, n in alloc.items():
        part = df[df[key] == grp].sample(n=min(n, len(df[df[key] == grp])), replace=False, random_state=42)
        parts.append(part)

    sampled = pd.concat(parts, ignore_index=True)
    return sampled


def compute_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 256):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False)
        embeddings.extend(embs)
    return embeddings


def build_index(
    input_csv: str,
    persist_dir: str = "vector_store",
    collection_name: str = "complaints",
    target_n: int = 12000,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    df = load_filtered_dataset(input_csv)
    ensure_columns(df)

    sample = stratified_sample(df, target_n=target_n)
    print(f"Stratified sample size: {len(sample)}")

    rows = sample.to_dict(orient="records")
    chunks = chunk_with_metadata(rows, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No chunks produced from sample")

    texts = [c["text"] for c in chunks]
    print(f"Total chunks to embed: {len(texts)}")

    embeddings = compute_embeddings(texts)

    # build chroma collection
    try:
        import chromadb
        from chromadb.config import Settings

        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
        client = chromadb.Client(settings=settings)
    except Exception:
        # fallback to default client
        import chromadb

        client = chromadb.Client()

    # create or get collection
    if collection_name in [c.name for c in client.list_collections()]:
        col = client.get_collection(collection_name)
        print(f"Replacing existing collection: {collection_name}")
        client.delete_collection(collection_name)

    col = client.create_collection(name=collection_name)

    ids = [f"{c['complaint_id']}_chunk{c['chunk_index']}" for c in chunks]
    metadatas = [
        {"complaint_id": c["complaint_id"], "product_category": c["product_category"]}
        for c in chunks
    ]
    documents = [c["text"] for c in chunks]

    # add in batches to avoid memory spikes
    batch = 5000
    for i in range(0, len(documents), batch):
        j = i + batch
        col.add(
            ids=ids[i:j],
            metadatas=metadatas[i:j],
            documents=documents[i:j],
            embeddings=embeddings[i:j],
        )

    client.persist()
    print(f"Index persisted to: {persist_dir} (collection: {collection_name})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to filtered_complaints.csv")
    parser.add_argument("--persist_dir", default="vector_store")
    parser.add_argument("--target_n", type=int, default=12000)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    args = parser.parse_args()

    build_index(
        input_csv=args.input,
        persist_dir=args.persist_dir,
        target_n=args.target_n,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
