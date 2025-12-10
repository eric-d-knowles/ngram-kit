"""Core filtering pipeline for Davies corpus."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, Set, Any
from collections import defaultdict

from ngramkit.common_db.api import open_db
from ngramkit.ngram_pivot.encoding import decode_year_ngram_key, encode_year_ngram_key, encode_year_stats, decode_year_stats
from ngramkit.utilities.display import format_banner, format_bytes

from ..davies_acquire.encoding import decode_occurrence_count
from .config import FilterConfig, PipelineConfig
from .filters.processor_factory import build_processor

logger = logging.getLogger(__name__)

__all__ = ["filter_davies_corpus"]


def filter_davies_corpus(
    src_db_path: str | Path,
    dst_db_path: str | Path,
    filter_config: Optional[FilterConfig] = None,
    # Filter configuration parameters (used if filter_config not provided)
    stop_set: Optional[Set[str]] = None,
    lemma_gen: Any = None,
    lowercase: Optional[bool] = None,
    alpha_only: Optional[bool] = None,
    filter_short: Optional[bool] = None,
    filter_stops: Optional[bool] = None,
    apply_lemmatization: Optional[bool] = None,
    min_len: Optional[int] = None,
    # Processing parameters
    write_batch_size: int = 100_000,
) -> None:
    """
    Filter Davies corpus database (single-threaded for simplicity).

    Reads sentences from raw database, applies filters, writes to filtered database.

    Args:
        src_db_path: Path to source (raw) database
        dst_db_path: Path to destination (filtered) database
        filter_config: Filtering configuration (uses defaults if not provided)
        stop_set: Set of stopwords to filter (used if filter_config not provided)
        lemma_gen: Lemmatizer instance (used if filter_config not provided)
        lowercase: Apply lowercasing (used if filter_config not provided)
        alpha_only: Filter non-alphabetic tokens (used if filter_config not provided)
        filter_short: Filter short tokens (used if filter_config not provided)
        filter_stops: Filter stopwords (used if filter_config not provided)
        apply_lemmatization: Apply lemmatization (used if filter_config not provided)
        min_len: Minimum token length (used if filter_config not provided)
        write_batch_size: Number of sentences per batch write
    """
    logger.info("Starting Davies corpus filtering pipeline")
    start_time = datetime.now()

    # Paths
    src_db_path = Path(src_db_path)
    dst_db_path = Path(dst_db_path)

    if not src_db_path.exists():
        raise ValueError(f"Source database does not exist: {src_db_path}")

    # Handle existing destination database - always remove for fresh start
    if dst_db_path.exists():
        logger.info("Removing existing destination database")
        from ngramkit.ngram_acquire.utils.cleanup import safe_db_cleanup
        if not safe_db_cleanup(dst_db_path):
            raise RuntimeError(
                f"Failed to remove existing database at {dst_db_path}. "
                "Close open handles or remove it manually."
            )
        logger.info("Successfully removed existing database")

    # Ensure parent directory exists
    dst_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Construct FilterConfig if not provided
    if filter_config is None:
        # If filter parameters provided, use them; otherwise use defaults
        if any(param is not None for param in [stop_set, lemma_gen, lowercase, alpha_only,
                                                 filter_short, filter_stops, apply_lemmatization, min_len]):
            kwargs = {}
            if stop_set is not None:
                kwargs['stop_set'] = stop_set
            if lemma_gen is not None:
                kwargs['lemma_gen'] = lemma_gen
            if lowercase is not None:
                kwargs['lowercase'] = lowercase
            if alpha_only is not None:
                kwargs['alpha_only'] = alpha_only
            if filter_short is not None:
                kwargs['filter_short'] = filter_short
            if filter_stops is not None:
                kwargs['filter_stops'] = filter_stops
            if apply_lemmatization is not None:
                kwargs['apply_lemmatization'] = apply_lemmatization
            if min_len is not None:
                kwargs['min_len'] = min_len
            filter_config = FilterConfig(**kwargs)
        else:
            filter_config = FilterConfig()

    # Build processor
    processor = build_processor(filter_config)

    # Print header
    print(format_banner("DAVIES CORPUS FILTERING", style="━"))
    print(f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}")
    print()
    print(format_banner("Configuration"))
    print(f"Source DB:            {src_db_path}")
    print(f"Destination DB:       {dst_db_path}")
    print(f"Lowercase:            {filter_config.lowercase}")
    print(f"Alpha only:           {filter_config.alpha_only}")
    print(f"Filter short:         {filter_config.filter_short} (min_len={filter_config.min_len})")
    print(f"Filter stops:         {filter_config.filter_stops}")
    print(f"Apply lemmas:         {filter_config.apply_lemmatization}")
    print(f"Batch size:           {write_batch_size:,}")
    print()
    print(format_banner("Processing Sentences"))
    sys.stdout.flush()

    # Process database
    sentences_read = 0
    unique_sentences_written = 0
    sentences_rejected = 0

    # Open both databases
    with open_db(src_db_path, mode="r", profile="read:packed24") as src_db, \
         open_db(dst_db_path, mode="w", profile="write:packed24", create_if_missing=True) as dst_db:

        # Track unique sentences written (approximate - counted when first seen)
        seen_sentences = set()

        # Iterate through all entries
        iterator = src_db.iterator()
        iterator.seek(b"")

        while iterator.valid():
            # Get key and value from iterator
            key_bytes = iterator.key()
            value_bytes = iterator.value()

            sentences_read += 1

            # Decode key to get year and sentence
            year, sentence_bytes = decode_year_ngram_key(key_bytes)

            # Decode occurrence count from value (16-byte format)
            count = decode_occurrence_count(value_bytes)

            # Apply filter
            filtered_sentence = processor(sentence_bytes)

            if filtered_sentence is None:
                sentences_rejected += 1
            else:
                # Create new key with filtered sentence
                new_key = encode_year_ngram_key(year, filtered_sentence)

                # Use packed24 merge operator by storing (year, occurrences, 1) as 24-byte value
                # The merge operator will automatically sum occurrences across duplicates
                new_value = encode_year_stats(count, 1)

                # Use merge() instead of put() to accumulate counts for duplicates
                dst_db.merge(new_key, new_value)

                # Track unique sentences for reporting
                if new_key not in seen_sentences:
                    seen_sentences.add(new_key)
                    unique_sentences_written += 1

            # Print progress
            if sentences_read % 500_000 == 0:
                print(f"  Processed {sentences_read:,} sentences "
                      f"({unique_sentences_written:,} unique seen, {sentences_rejected:,} rejected)")
                sys.stdout.flush()

            # Move to next entry
            iterator.next()

    # Print completion summary
    end_time = datetime.now()
    elapsed = end_time - start_time

    print("\nProcessing complete!")
    print()
    print(format_banner("Final Summary"))
    print(f"Sentences read:           {sentences_read:,}")
    print(f"Unique written:           {unique_sentences_written:,}")
    print(f"Sentences rejected:       {sentences_rejected:,}")
    retention_pct = (unique_sentences_written / sentences_read * 100) if sentences_read > 0 else 0
    print(f"Retention rate:           {retention_pct:.1f}%")
    print(f"Destination DB:           {dst_db_path}")
    print()
    print(f"End Time: {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"Total Runtime: {elapsed}")
    print()

    logger.info(f"Filtering complete: {unique_sentences_written:,} unique sentences written")
