"""Encoding utilities for Davies corpus database format.

Davies corpora are stored in the same pivoted format as ngrams:
    Key: (year, sentence_tokens)
    Value: () or dummy value (no frequency data)

This allows the same training code to work for both corpora.
"""
from __future__ import annotations

from typing import List

# Import encoding utilities from ngramkit
from ngramkit.ngram_pivot.encoding import (
    encode_year_ngram_key,
    decode_year_ngram_key,
    encode_year_stats,
    decode_year_stats,
)

__all__ = [
    "encode_sentence_key",
    "decode_sentence_key",
    "encode_occurrence_count",
    "decode_occurrence_count",
]


def encode_occurrence_count(count: int, year: int = 0) -> bytes:
    """
    Encode occurrence count as 24-byte value for packed24 merge operator.

    The packed24 merge operator expects 24-byte records: (year, occurrences, documents).
    We store (year, occurrences, 1) where year is redundant with the key but needed for merge.

    Args:
        count: Number of occurrences
        year: Year value (optional, defaults to 0)

    Returns:
        24-byte packed value
    """
    import struct
    return struct.pack('<QQQ', year, count, 1)


def decode_occurrence_count(value: bytes) -> int:
    """
    Decode occurrence count from packed value.

    Handles both 16-byte (old format) and 24-byte (new format with year) values.

    Args:
        value: Packed value (16 or 24 bytes)

    Returns:
        Number of occurrences
    """
    import struct
    if len(value) == 24:
        # New format: (year, occurrences, documents)
        year, occurrences, documents = struct.unpack('<QQQ', value)
        return occurrences
    elif len(value) == 16:
        # Old format: (occurrences, documents)
        occurrences, documents = decode_year_stats(value)
        return occurrences
    else:
        raise ValueError(f"Unexpected value length: {len(value)} bytes (expected 16 or 24)")


def encode_sentence_key(year: int, tokens: List[str]) -> bytes:
    """
    Encode a sentence into a database key.

    Uses the same format as pivoted ngrams: (year, tokens).

    Args:
        year: Year for this sentence
        tokens: List of word tokens

    Returns:
        Encoded key bytes

    Example:
        >>> tokens = ["the", "cat", "sat"]
        >>> key = encode_sentence_key(1950, tokens)
        >>> isinstance(key, bytes)
        True
    """
    # Join tokens with spaces to create the "ngram"
    sentence_bytes = ' '.join(tokens).encode('utf-8')

    # Use ngram encoding from ngramkit
    return encode_year_ngram_key(year, sentence_bytes)


def decode_sentence_key(key: bytes) -> tuple[int, List[str]]:
    """
    Decode a database key into year and tokens.

    Args:
        key: Encoded key bytes

    Returns:
        Tuple of (year, tokens)

    Example:
        >>> key = encode_sentence_key(1950, ["the", "cat", "sat"])
        >>> year, tokens = decode_sentence_key(key)
        >>> year
        1950
        >>> tokens
        ['the', 'cat', 'sat']
    """
    # Decode using ngram encoding
    year, sentence_bytes = decode_year_ngram_key(key)

    # Split back into tokens
    tokens = sentence_bytes.decode('utf-8').split()

    return year, tokens
