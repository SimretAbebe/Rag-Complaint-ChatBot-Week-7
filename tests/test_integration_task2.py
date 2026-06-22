import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


def test_task2_integration_monkeypatched(monkeypatch):
    # Prepare tiny CSV
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "small_complaints.csv"
        df = pd.DataFrame({
            'complaint_id': [1, 2, 3],
            'product_category': ['A', 'B', 'A'],
            'complaint_text': ['foo', 'bar', 'baz']
        })
        df.to_csv(csv_path, index=False)

        persist_dir = Path(td) / 'persist'

        # Monkeypatch SentenceTransformer used in the script to a dummy
        import scripts.task2_build_index as builder

        class DummyModel:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, batch, show_progress_bar=False, convert_to_numpy=True):
                # return fixed-dimension vectors
                import numpy as np

                return np.array([[len(t), len(t)] for t in batch], dtype=float)

        monkeypatch.setattr(builder, 'SentenceTransformer', DummyModel)

        # Monkeypatch chromadb.Client used in the script to a dummy client
        class DummyCollection:
            def add(self, ids, documents, embeddings, metadatas):
                # simple validation
                assert len(ids) == len(documents) == len(embeddings) == len(metadatas)

        class DummyClient:
            def __init__(self, settings=None):
                pass

            def get_or_create_collection(self, name, metadata=None):
                return DummyCollection()

            def persist(self):
                return None

        monkeypatch.setattr(builder, 'chromadb', builder.chromadb)
        monkeypatch.setattr(builder.chromadb, 'Client', lambda settings=None: DummyClient())

        # Run the script's main with argv pointing to our CSV and persist_dir
        monkeypatch.setattr(sys, 'argv', ['task2_build_index.py', '--input', str(csv_path), '--persist_dir', str(persist_dir)])
        # Call main and ensure manifest is written
        builder.main()

        manifest_path = persist_dir / 'index_manifest.json'
        assert manifest_path.exists(), 'index_manifest.json should be created'
        manifest = json.loads(manifest_path.read_text(encoding='utf8'))
        assert manifest['rows_indexed'] == 3