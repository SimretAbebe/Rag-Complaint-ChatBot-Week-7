import pandas as pd
from scripts.task2_build_index import stratified_sample
from scripts.task2_build_index import embed_texts, main as build_main
import sys
import numpy as np


def test_stratified_sample_fraction():
    df = pd.DataFrame({
        'complaint_id': list(range(100)),
        'product_category': ['A'] * 50 + ['B'] * 30 + ['C'] * 20,
        'complaint_text': ['x'] * 100
    })

    sampled = stratified_sample(df, strat_col='product_category', sample_frac=0.2, seed=0)
    # expect approx 20% of each group
    counts = sampled['product_category'].value_counts().to_dict()
    assert counts['A'] in (10, 9, 11)
    assert counts['B'] in (6, 5, 7)
    assert counts['C'] in (4, 3, 5)


def test_stratified_sample_n_per_group():
    df = pd.DataFrame({
        'complaint_id': list(range(60)),
        'product_category': ['A'] * 30 + ['B'] * 20 + ['C'] * 10,
        'complaint_text': ['x'] * 60
    })

    sampled = stratified_sample(df, strat_col='product_category', n_per_group=5, seed=1)
    counts = sampled['product_category'].value_counts().to_dict()
    assert counts['A'] == 5
    assert counts['B'] == 5
    assert counts['C'] == 5


def test_embed_texts_with_dummy_model():
    class DummyModel:
        def encode(self, batch, show_progress_bar=False, convert_to_numpy=True):
            # return a deterministic numpy array per batch
            return np.array([[len(t)] * 8 for t in batch], dtype=float)

    texts = ["short", "a bit longer text", "x"]
    model = DummyModel()
    embeddings = embed_texts(model, texts, batch_size=2)
    assert len(embeddings) == 3
    assert len(embeddings[0]) == 8
    assert embeddings[0][0] == len(texts[0])


def test_main_raises_on_missing_input(monkeypatch):
    # simulate running the script with a missing input file
    monkeypatch.setattr(sys, 'argv', ['task2_build_index.py', '--input', 'nonexistent_file.csv'])
    try:
        build_main()
    except FileNotFoundError:
        # expected
        return
    except SystemExit:
        # argparse may call SystemExit; ensure it's treated as failure
        return
    raise AssertionError('Expected FileNotFoundError or SystemExit when input missing')
