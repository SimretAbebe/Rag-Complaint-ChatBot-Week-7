import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import chromadb
from chromadb.config import Settings


def stratified_sample(df: pd.DataFrame, strat_col: str, sample_frac: float = None, n_per_group: int = None, seed: int = 42) -> pd.DataFrame:
    if sample_frac is not None:
        # use sklearn's train_test_split with stratify for reproducible fraction
        train, _ = train_test_split(df, train_size=sample_frac, stratify=df[strat_col], random_state=seed)
        return train.reset_index(drop=True)

    if n_per_group is not None:
        # sample up to `n_per_group` rows per group and concat to ensure a DataFrame
        groups = []
        for _, g in df.groupby(strat_col):
            groups.append(g.sample(n=min(len(g), n_per_group), random_state=seed))
        if groups:
            sampled = pd.concat(groups, axis=0, ignore_index=True)
        else:
            sampled = df.iloc[0:0].copy()
        return sampled.reset_index(drop=True)

    return df.reset_index(drop=True)


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 128, device: str = None) -> List[List[float]]:
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)
    if embeddings:
        return np.vstack(embeddings).tolist()
    return []


def build_chroma_index(df: pd.DataFrame, embeddings, persist_dir: str, collection_name: str = "complaints"):
    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
    client = chromadb.Client(settings)
    collection = client.get_or_create_collection(name=collection_name, metadata={"source": "filtered_complaints"})

    ids = df['complaint_id'].astype(str).tolist()
    documents = df['complaint_text'].astype(str).tolist()
    metadatas = df[['complaint_id', 'product_category']].rename(columns={'complaint_id': 'complaint_id', 'product_category': 'product_category'}).to_dict(orient='records')

    # Remove existing collection items with same ids (idempotent add)
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    client.persist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/processed/filtered_complaints.csv')
    parser.add_argument('--persist_dir', type=str, default='vector_store/chroma_store')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--sample_frac', type=float, default=None)
    parser.add_argument('--n_per_group', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    # Expect columns: complaint_id, product_category, complaint_text (or combine)
    if 'complaint_text' not in df.columns:
        # try combining available text columns
        text_cols = [c for c in df.columns if 'text' in c or 'complaint' in c.lower()]
        if text_cols:
            df['complaint_text'] = df[text_cols[0]].astype(str)
        else:
            raise ValueError('No complaint text column found in CSV')

    sampled = stratified_sample(df, strat_col='product_category', sample_frac=args.sample_frac, n_per_group=args.n_per_group, seed=args.seed)
    print(f"Building embeddings for {len(sampled):,} records (model={args.model})")

    model = SentenceTransformer(args.model)
    embeddings = embed_texts(model, sampled['complaint_text'].astype(str).tolist(), batch_size=args.batch_size)

    os.makedirs(args.persist_dir, exist_ok=True)
    build_chroma_index(sampled, embeddings, persist_dir=args.persist_dir)

    manifest = {
        'persist_dir': str(args.persist_dir),
        'model': args.model,
        'rows_indexed': len(sampled),
        'seed': args.seed
    }
    with open(Path(args.persist_dir) / 'index_manifest.json', 'w', encoding='utf8') as f:
        json.dump(manifest, f, indent=2)

    print('Index build complete. Manifest written to', Path(args.persist_dir) / 'index_manifest.json')


if __name__ == '__main__':
    main()
