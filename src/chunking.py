from typing import List, Dict, Iterable

# Import RecursiveCharacterTextSplitter from LangChain when available,
# otherwise fall back to the local test-friendly shim `langchain_text_splitters`.
try:
    # newer langchain uses `text_splitters`
    from langchain.text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except Exception:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:
        # As a final fallback, provide a minimal splitter compatible with our usage.
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=500, chunk_overlap=50, **kwargs):
                self.chunk_size = int(chunk_size)
                self.chunk_overlap = int(chunk_overlap)

            def split_text(self, text: str) -> List[str]:
                s = "" if text is None else str(text)
                n = len(s)
                if n <= self.chunk_size:
                    return [s]
                chunks: List[str] = []
                step = max(1, self.chunk_size - self.chunk_overlap)
                i = 0
                while i < n:
                    chunks.append(s[i : i + self.chunk_size])
                    i += step
                return chunks


def chunk_texts(texts: Iterable[str], chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Chunk an iterable of texts into smaller passages using a character splitter.

    Returns a flat list of chunks corresponding to input texts order.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_chunks: List[str] = []
    for t in texts:
        if t is None:
            continue
        if not isinstance(t, str):
            try:
                t = str(t)
            except Exception:
                continue
        if t.strip() == "":
            continue
        chunks = splitter.split_text(t)
        all_chunks.extend(chunks)
    return all_chunks


def chunk_with_metadata(rows: Iterable[Dict], *, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """Given iterable of dicts with keys `complaint_id`, `product_category`, `cleaned_narrative`,
    produce a list of chunk dicts: {id, complaint_id, product_category, text, chunk_index}.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    result: List[Dict] = []
    for row in rows:
        text = row.get("cleaned_narrative")
        if text is None:
            continue
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                continue
        if text.strip() == "":
            continue
        chunks = splitter.split_text(text)
        for i, c in enumerate(chunks):
            result.append(
                {
                    "complaint_id": row.get("complaint_id"),
                    "product_category": row.get("product_category"),
                    "text": c,
                    "chunk_index": i,
                }
            )
    return result
