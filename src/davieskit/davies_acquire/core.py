"""Main entry point for Davies corpus acquisition pipeline."""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

from ngramkit.common_db.api import open_db
from ngramkit.utilities.display import format_banner, format_bytes

from .reader import discover_text_files, extract_year_from_filename, read_text_file
from .tokenizer import tokenize_sentences
from .writer import SentenceBatchWriter

logger = logging.getLogger(__name__)

__all__ = ["ingest_davies_corpus"]


def process_single_file(
    zip_path: Path,
    year: int,
) -> Tuple[str, int, int]:
    """
    Process a single text file: read, tokenize, return sentences.

    This function runs in a worker process.

    Args:
        zip_path: Path to zip file
        year: Year for this file

    Returns:
        Tuple of (filename, sentence_count, error_count)
    """
    sentence_count = 0
    error_count = 0

    try:
        # Read documents from zip file
        for doc_year, text in read_text_file(zip_path, year):
            try:
                # Tokenize into sentences
                for tokens in tokenize_sentences(text):
                    sentence_count += 1
            except Exception as e:
                logger.warning(f"Error tokenizing document in {zip_path.name}: {e}")
                error_count += 1

    except Exception as e:
        logger.error(f"Error processing {zip_path.name}: {e}")
        error_count += 1

    return zip_path.name, sentence_count, error_count


def _perform_compaction(db, db_path: Path) -> None:
    """
    Perform full compaction on the database using compact_all().

    Args:
        db: Open RocksDB handle
        db_path: Path to database (for logging)
    """
    logger.info("Starting post-ingestion compaction")
    print()
    print(format_banner("Post-Ingestion Compaction"))

    # Get initial size if possible
    try:
        initial_size = db.get_property("rocksdb.total-sst-files-size")
        initial_size = int(initial_size) if initial_size else None
        if initial_size:
            print(f"Initial DB size:         {format_bytes(initial_size)}")
    except Exception:
        initial_size = None

    sys.stdout.flush()

    start_time = time.time()
    try:
        db.compact_all()
        elapsed = time.time() - start_time

        print(f"Compaction completed in {timedelta(seconds=int(elapsed))}")

        # Get final size if possible
        try:
            final_size = db.get_property("rocksdb.total-sst-files-size")
            final_size = int(final_size) if final_size else None
            if initial_size and final_size:
                saved = initial_size - final_size
                pct = (saved / initial_size) * 100
                print(f"Size before:             {format_bytes(initial_size)}")
                print(f"Size after:              {format_bytes(final_size)}")
                print(f"Space saved:             {format_bytes(saved)} ({pct:.1f}%)")
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Compaction failed: {e}")
        print(f"Compaction failed: {e}")
        print("Database is still usable, but may not be optimally compacted.")


def ingest_davies_corpus(
    db_path_stub: str,
    workers: Optional[int] = None,
    write_batch_size: int = 100_000,
    compact_after_ingest: bool = False,
) -> None:
    """
    Main pipeline: read Davies corpus text files and ingest into RocksDB.

    Orchestrates the complete Davies acquisition workflow:
    1. Discovers text files in corpus directory
    2. Opens/creates RocksDB in pivoted format
    3. Reads and tokenizes text files
    4. Writes sentences directly to pivoted DB: (year, tokens) -> ()
    5. Optionally performs post-ingestion compaction

    Args:
        db_path_stub: Base path containing corpus name (e.g., "/path/to/COHA")
        workers: Number of concurrent workers (default: cpu_count - 1)
        write_batch_size: Number of sentences per batch write
        compact_after_ingest: If True, perform full compaction after ingestion
    """
    logger.info("Starting Davies corpus acquisition pipeline")
    start_time = datetime.now()

    # Extract corpus name from db_path_stub
    db_path_stub_obj = Path(db_path_stub)
    corpus_name = db_path_stub_obj.name

    # Corpus path is the same as db_path_stub
    corpus_path = db_path_stub_obj

    # Database path is db_path_stub/db
    db_path = db_path_stub_obj / "db"

    # Validate corpus path
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise ValueError(f"Corpus path does not exist: {corpus_path}")

    text_dir = corpus_path / "text"
    if not text_dir.exists():
        raise ValueError(f"Text directory not found: {text_dir}")

    # Handle existing database - always remove for fresh start
    db_path = Path(db_path)
    if db_path.exists():
        logger.info("Removing existing database for fresh start")
        # Use safe cleanup from ngramkit
        from ngramkit.ngram_acquire.utils.cleanup import safe_db_cleanup
        if not safe_db_cleanup(db_path):
            raise RuntimeError(
                f"Failed to remove existing database at {db_path}. "
                "Close open handles or remove it manually."
            )
        logger.info("Successfully removed existing database")

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine worker count
    if workers is None:
        cpu_count = os.cpu_count() or 4
        workers = max(1, cpu_count - 1)

    # Discover text files
    logger.info("Discovering text files...")
    text_files = discover_text_files(text_dir)

    # Extract years from filenames
    file_year_pairs: List[Tuple[Path, int]] = []
    for text_file in text_files:
        try:
            year = extract_year_from_filename(text_file.name)
            file_year_pairs.append((text_file, year))
        except ValueError as e:
            logger.warning(f"Skipping file: {e}")
            continue

    # Print pipeline header
    print(format_banner(f"{corpus_name} CORPUS ACQUISITION", style="━"))
    print(f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}")
    print()
    print(format_banner("Configuration"))
    print(f"Corpus path:          {corpus_path}")
    print(f"Text directory:       {text_dir}")
    print(f"DB path:              {db_path}")
    print(f"Text files found:     {len(file_year_pairs)}")
    print(f"Workers:              {workers}")
    print(f"Batch size:           {write_batch_size:,}")
    print()
    print(format_banner("Processing Files"))
    sys.stdout.flush()

    # Open database and create writer
    logger.info("Opening database...")
    with open_db(db_path, profile="write:packed24", create_if_missing=True) as db:
        writer = SentenceBatchWriter(db, batch_size=write_batch_size)

        total_sentences = 0
        total_errors = 0
        files_processed = 0

        # Process files sequentially for now (easier to track progress)
        # TODO: Add parallel processing later
        for text_file, year in file_year_pairs:
            try:
                # Read and tokenize documents
                file_sentences = 0
                for doc_year, text in read_text_file(text_file, year):
                    for tokens in tokenize_sentences(text):
                        writer.add(year, tokens)
                        file_sentences += 1

                total_sentences += file_sentences
                files_processed += 1

                # Print progress every 5 files
                if files_processed % 5 == 0:
                    print(f"  Processed {files_processed}/{len(file_year_pairs)} files "
                          f"({total_sentences:,} sentences)")
                    sys.stdout.flush()

            except Exception as e:
                logger.error(f"Error processing {text_file.name}: {e}")
                total_errors += 1
                continue

        # Flush remaining sentences
        logger.info("Flushing remaining sentences...")
        writer.close()

        # Optional post-ingestion compaction
        if compact_after_ingest:
            _perform_compaction(db, db_path)

    # Print completion summary
    end_time = datetime.now()
    elapsed = end_time - start_time

    print("\nProcessing complete!")
    print()
    print(format_banner("Final Summary"))
    print(f"Files processed:          {files_processed}/{len(file_year_pairs)}")
    print(f"Failed files:             {total_errors}")
    print(f"Total sentences written:  {total_sentences:,}")
    print(f"Database path:            {db_path}")
    print()
    print(f"End Time: {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"Total Runtime: {elapsed}")
    print()

    logger.info(f"Acquisition complete: {total_sentences:,} sentences written")
