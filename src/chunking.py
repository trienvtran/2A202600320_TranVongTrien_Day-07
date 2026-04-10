from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks

class SentenceChunker:
    def __init__(self, max_sentences_per_chunk: int = 4, overlap_sentences: int = 1) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)
        self.overlap_sentences = min(overlap_sentences, self.max_sentences_per_chunk - 1)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Tách câu
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        # Bước nhảy = Kích thước chunk - Số câu trùng lặp
        step = self.max_sentences_per_chunk - self.overlap_sentences
        
        for i in range(0, len(sentences), step):
            batch = sentences[i : i + self.max_sentences_per_chunk]
            chunk_content = " ".join(batch).strip()
            if chunk_content:
                chunks.append(chunk_content)
                
            # Dừng lại nếu đã lấy đến cuối danh sách câu
            if i + self.max_sentences_per_chunk >= len(sentences):
                break
        
        return chunks

class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        """Chunk text using recursive splitting strategy."""
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        """Recursively split text using separators in order."""
        if not current_text:
            return []
        
        # If text is within chunk size, return as single chunk
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        # If no separators left, force split at chunk_size
        if not remaining_separators:
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunk = current_text[i : i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks if chunks else [current_text]
        
        # Try first separator
        separator = remaining_separators[0]
        remaining = remaining_separators[1:]
        
        # Split by separator
        if separator:
            parts = current_text.split(separator)
        else:
            # Empty separator means split into characters
            parts = list(current_text)
        
        # Recursively process parts
        chunks = []
        for part in parts:
            if part:
                if len(part) <= self.chunk_size:
                    chunks.append(part)
                else:
                    # Part is too large, recurse with remaining separators
                    sub_chunks = self._split(part, remaining)
                    chunks.extend(sub_chunks)
        
        return chunks if chunks else [current_text]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # Compute dot product
    dot_product = _dot(vec_a, vec_b)
    
    # Compute magnitudes
    magnitude_a = math.sqrt(sum(x * x for x in vec_a)) or 0.0
    magnitude_b = math.sqrt(sum(x * x for x in vec_b)) or 0.0
    
    # Return 0 if either has zero magnitude
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        """Compare three chunking strategies on the given text."""
        result = {}
        
        # Strategy 1: Fixed Size
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=0)
        fixed_chunks = fixed_chunker.chunk(text)
        result['fixed_size'] = {
            'count': len(fixed_chunks),
            'avg_length': sum(len(c) for c in fixed_chunks) / len(fixed_chunks) if fixed_chunks else 0,
            'chunks': fixed_chunks
        }
        
        # Strategy 2: By Sentences
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        sentence_chunks = sentence_chunker.chunk(text)
        result['by_sentences'] = {
            'count': len(sentence_chunks),
            'avg_length': sum(len(c) for c in sentence_chunks) / len(sentence_chunks) if sentence_chunks else 0,
            'chunks': sentence_chunks
        }
        
        # Strategy 3: Recursive
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)
        recursive_chunks = recursive_chunker.chunk(text)
        result['recursive'] = {
            'count': len(recursive_chunks),
            'avg_length': sum(len(c) for c in recursive_chunks) / len(recursive_chunks) if recursive_chunks else 0,
            'chunks': recursive_chunks
        }
        
        return result
