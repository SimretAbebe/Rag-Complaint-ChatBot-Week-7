import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.skipif(os.getenv('RUN_INTEGRATION_TESTS') != '1', reason='Integration tests disabled')
def test_chroma_persists_files(monkeypatch):
    # Create a tiny CSV and run the builder with a dummy model, using real chromadb persistence
    from scripts import task2_build_index as builder

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "small.csv"
        df = pd.DataFrame({
            'complaint_id': [10, 11],
            'product_category': ['X', 'Y'],
            'complaint_text': ['a', 'bb']
        })
        df.to_csv(csv_path, index=False)

        persist_dir = Path(td) / 'chroma_store'

        # Dummy model to avoid heavy downloads
        class DummyModel:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, batch, show_progress_bar=False, convert_to_numpy=True):
                import numpy as np

                return np.array([[len(t)] * 4 for t in batch], dtype=float)

        monkeypatch.setattr(builder, 'SentenceTransformer', DummyModel)

        # Run builder
        monkeypatch.setattr(sys, 'argv', ['task2_build_index.py', '--input', str(csv_path), '--persist_dir', str(persist_dir)])
        builder.main()

        # assert persist_dir exists and has files (non-empty)
        assert persist_dir.exists(), 'persist_dir should exist'
        files = list(persist_dir.rglob('*'))
        # At least one file should be created by chromadb persistence
        assert any(f.is_file() for f in files), 'No persisted files found in persist_dir'