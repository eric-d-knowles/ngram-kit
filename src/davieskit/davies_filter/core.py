"""Core filtering pipeline for Davies corpus."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from ngramkit.common_db.api import open_db
from ngramkit.ngram_pivot.encoding import decode_year_ngram_key, encode_year_ngram_key
from ngramkit.utilities.display import format_banner, format_bytes

from .config import FilterConfig, PipelineConfig
from .filters.processor_factory import build_processor

logger = logging.getLogger(__name__)

__all__ = ["filter_davies_corpus"]


def filter_davies_corpus(
    src_db_path: str | Path,
    dst_db_path: str | Path,
    filter_config: Optional[FilterConfig] = None,
    overwrite_db: bool = True,
    write_batch_size: int = 100_000,
) -> None:
    """
    Filter Davies corpus database (single-threaded for simplicity).

    Reads sentences from raw database, applies filters, writes to filtered database.

    Args:
        src_db_path: Path to source (raw) database
        dst_db_path: Path to destination (filtered) database
        filter_config: Filtering configuration (uses defaults if not provided)
        overwrite_db: Whether to remove existing destination DB
        write_batch_size: Number of sentences per batch write
    """
    logger.info("Starting Davies corpus filtering pipeline")
    start_time = datetime.now()

    # Paths
    src_db_path = Path(src_db_path)
    dst_db_path = Path(dst_db_path)

    if not src_db_path.exists():
        raise ValueError(f"Source database does not exist: {src_db_path}")

    # Handle existing destination database
    if overwrite_db and dst_db_path.exists():
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

    # Use default config if not provided
    if filter_config is None:
        filter_config = FilterConfig()

    # Build processor
    processor = build_processor(filter_config)

    # Print header
    print()
    print(format_banner("DAVIES CORPUS FILTERING"))
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuration")
    print("=" * 80)
    print(f"Source DB:          {src_db_path}")
    print(f"Destination DB:     {dst_db_path}")
    print(f"Lowercase:          {filter_config.lowercase}")
    print(f"Alpha only:         {filter_config.alpha_only}")
    print(f"Filter short:       {filter_config.filter_short} (min_len={filter_config.min_len})")
    print(f"Filter stops:       {filter_config.filter_stops}")
    print(f"Apply lemmas:       {filter_config.apply_lemmatization}")
    print(f"Batch size:         {write_batch_size:,}")
    print()
    sys.stdout.flush()

    # Process database
    sentences_read = 0
    sentences_written = 0
    sentences_rejected = 0

    print("Processing sentences...")
    print("=" * 80)
    sys.stdout.flush()

    # Open both databases
    with open_db(src_db_path, mode="r", profile="read:packed24") as src_db, \
         open_db(dst_db_path, mode="w", profile="write:packed24", create_if_missing=True) as dst_db:

        # Prepare write batch
        batch_buffer = []

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

            # Apply filter
            filtered_sentence = processor(sentence_bytes)

            if filtered_sentence is None:
                sentences_rejected += 1
            else:
                # Create new key with filtered sentence
                new_key = encode_year_ngram_key(year, filtered_sentence)
                batch_buffer.append((new_key, value_bytes))
                sentences_written += 1

            # Write batch when buffer is full
            if len(batch_buffer) >= write_batch_size:
                with dst_db.write_batch(disable_wal=True, sync=False) as wb:
                    for k, v in batch_buffer:
                        wb.put(k, v)
                batch_buffer.clear()

                # Print progress
                if sentences_read % 500_000 == 0:
                    print(f"  Processed {sentences_read:,} sentences "
                          f"({sentences_written:,} written, {sentences_rejected:,} rejected)")
                    sys.stdout.flush()

            # Move to next entry
            iterator.next()

        # Write remaining batch
        if batch_buffer:
            with dst_db.write_batch(disable_wal=True, sync=False) as wb:
                for k, v in batch_buffer:
                    wb.put(k, v)

    # Print completion summary
    end_time = datetime.now()
    elapsed = end_time - start_time

    print()
    print(format_banner("FILTERING COMPLETE"))
    print(f"End Time:           {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:           {timedelta(seconds=int(elapsed.total_seconds()))}")
    print()
    print("Statistics")
    print("=" * 80)
    print(f"Sentences read:     {sentences_read:,}")
    print(f"Sentences written:  {sentences_written:,}")
    print(f"Sentences rejected: {sentences_rejected:,}")
    retention_pct = (sentences_written / sentences_read * 100) if sentences_read > 0 else 0
    print(f"Retention rate:     {retention_pct:.1f}%")
    print(f"Destination DB:     {dst_db_path}")
    print()

    # Get database size
    try:
        with open_db(dst_db_path, mode="r") as db:
            prop = db.get_property("rocksdb.total-sst-files-size")
            if prop:
                size_bytes = int(prop)
                print(f"Database size:      {format_bytes(size_bytes)}")
    except Exception:
        pass

    logger.info(f"Filtering complete: {sentences_written:,} sentences written")
    print()
