# tests/test_chunking.py

from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_splitter(chunk_size=500, chunk_overlap=50):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def test_chunking_produces_chunks():
    """Test that a long text produces multiple chunks"""
    splitter = get_text_splitter()
    long_text = "word " * 300  # 1500 characters
    chunks = splitter.split_text(long_text)
    assert len(chunks) > 1


def test_chunk_size_respected():
    """Test that no chunk exceeds the chunk size"""
    splitter = get_text_splitter(chunk_size=500)
    long_text = "word " * 300
    chunks = splitter.split_text(long_text)
    for chunk in chunks:
        assert len(chunk) <= 500


def test_short_text_stays_one_chunk():
    """Test that short text is not split"""
    splitter = get_text_splitter(chunk_size=500)
    short_text = "This is a short complaint about my credit card."
    chunks = splitter.split_text(short_text)
    assert len(chunks) == 1


def test_chunk_overlap_exists():
    """Test that consecutive chunks share some content"""
    splitter = get_text_splitter(chunk_size=100, chunk_overlap=20)
    text = "a " * 200
    chunks = splitter.split_text(text)
    if len(chunks) > 1:
        # Last part of chunk 1 should appear in start of chunk 2
        end_of_chunk1 = chunks[0][-20:]
        assert end_of_chunk1 in chunks[1]


def test_empty_text_handling():
    """Test that very short text is handled"""
    splitter = get_text_splitter()
    text = "short"
    chunks = splitter.split_text(text)
    assert len(chunks) >= 1


def test_stratified_sample_proportions():
    """Test that stratified sampling maintains proportions"""
    import pandas as pd
    import numpy as np

    # Create a mini fake dataset
    data = {
        'product_category': ['Credit Card'] * 400 +
                            ['Savings Account'] * 300 +
                            ['Money Transfer'] * 200 +
                            ['Personal Loan'] * 100,
        'cleaned_narrative': ['sample text'] * 1000
    }
    df = pd.DataFrame(data)

    # Apply stratified sampling
    SAMPLE_SIZE = 100
    total = len(df)
    sample_dfs = []

    for product, count in df['product_category'].value_counts().items():
        proportion = count / total
        sample_n = round(SAMPLE_SIZE * proportion)
        sampled = df[df['product_category'] == product].sample(
            n=sample_n, random_state=42
        )
        sample_dfs.append(sampled)

    df_sample = pd.concat(sample_dfs, ignore_index=True)

    # Check proportions are maintained within 5% tolerance
    for product, count in df['product_category'].value_counts().items():
        expected_prop = count / total
        actual_prop = len(df_sample[df_sample['product_category'] == product]) / len(df_sample)
        assert abs(expected_prop - actual_prop) < 0.05