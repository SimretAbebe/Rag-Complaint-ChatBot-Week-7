"""A minimal, dependency-free implementation of
`RecursiveCharacterTextSplitter` to support the test suite.

This is intentionally simple: it performs character-level chunking with
an overlap and exposes the same constructor arguments used in tests.
"""
from typing import Callable, List, Sequence


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        length_function: Callable[[str], int] = len,
        separators: Sequence[str] | None = None,
    ): 
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        s = "" if text is None else str(text)
        n = self.length_function(s)
        if n <= self.chunk_size:
            return [s]

        chunks: List[str] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        i = 0
        while i < n:
            chunk = s[i : i + self.chunk_size]
            chunks.append(chunk)
            i += step
        return chunks


__all__ = ["RecursiveCharacterTextSplitter"]
