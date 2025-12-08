"""Database writing utilities for Davies acquisition."""
from __future__ import annotations

from typing import List, Iterator, Tuple
import logging

from ngramkit.common_db.api import open_db
from .encoding import encode_sentence_key, EMPTY_VALUE

logger = logging.getLogger(__name__)

__all__ = [
    "write_sentences_batch",
    "SentenceBatchWriter",
]

DEFAULT_BATCH_SIZE = 100_000


def write_sentences_batch(
    db,
    sentences: List[Tuple[int, List[str]]],
    disable_wal: bool = True,
) -> int:
    """
    Write a batch of sentences to the database.

    Args:
        db: Open RocksDB handle
        sentences: List of (year, tokens) tuples
        disable_wal: If True, disable write-ahead log for better performance

    Returns:
        Number of sentences written

    Example:
        >>> with open_db(db_path, mode="w") as db:
        ...     count = write_sentences_batch(db, [
        ...         (1950, ["the", "cat", "sat"]),
        ...         (1951, ["on", "the", "mat"]),
        ...     ])
        >>> count
        2
    """
    if not sentences:
        return 0

    # Use context manager for batch writing
    with db.write_batch(disable_wal=disable_wal, sync=False) as wb:
        for year, tokens in sentences:
            key = encode_sentence_key(year, tokens)
            wb.put(key, EMPTY_VALUE)
    # Context manager auto-commits on exit

    return len(sentences)


class SentenceBatchWriter:
    """
    Buffered writer for sentences.

    Accumulates sentences in memory and writes them in batches
    for better performance.
    """

    def __init__(
        self,
        db,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize the batch writer.

        Args:
            db: Open RocksDB handle
            batch_size: Number of sentences per batch
        """
        self.db = db
        self.batch_size = batch_size
        self.buffer: List[Tuple[int, List[str]]] = []
        self.total_written = 0

    def add(self, year: int, tokens: List[str]) -> None:
        """
        Add a sentence to the buffer.

        Flushes automatically when buffer reaches batch_size.

        Args:
            year: Year for this sentence
            tokens: List of word tokens
        """
        self.buffer.append((year, tokens))

        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> int:
        """
        Write buffered sentences to database.

        Returns:
            Number of sentences written
        """
        if not self.buffer:
            return 0

        count = write_sentences_batch(self.db, self.buffer)
        self.total_written += count
        self.buffer.clear()

        return count

    def close(self) -> int:
        """
        Flush any remaining buffered sentences.

        Returns:
            Total number of sentences written
        """
        self.flush()
        return self.total_written
