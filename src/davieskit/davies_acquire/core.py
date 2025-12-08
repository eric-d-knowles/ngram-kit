"""Main entry point for Davies corpus acquisition pipeline."""
from __future__ import annotations

import logging
import os
import sys
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


def ingest_davies_corpus(
    corpus_name: str,
    corpus_path: str,
    db_path: str,
    overwrite_db: bool = True,
    workers: Optional[int] = None,
    write_batch_size: int = 100_000,
) -> None:
    """
    Main pipeline: read Davies corpus text files and ingest into RocksDB.

    Orchestrates the complete Davies acquisition workflow:
    1. Discovers text files in corpus directory
    2. Opens/creates RocksDB in pivoted format
    3. Reads and tokenizes text files
    4. Writes sentences directly to pivoted DB: (year, tokens) -> ()

    Args:
        corpus_name: Name of corpus (e.g., "COHA", "COCA")
        corpus_path: Path to corpus directory (containing text/ subdirectory)
        db_path: Path for output database
        overwrite_db: If True, remove existing database before starting
        workers: Number of concurrent workers (default: cpu_count - 1)
        write_batch_size: Number of sentences per batch write
    """
    logger.info("Starting Davies corpus acquisition pipeline")
    start_time = datetime.now()

    # Validate corpus path
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise ValueError(f"Corpus path does not exist: {corpus_path}")

    text_dir = corpus_path / "text"
    if not text_dir.exists():
        raise ValueError(f"Text directory not found: {text_dir}")

    # Handle existing database
    db_path = Path(db_path)
    if overwrite_db and db_path.exists():
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
    print()
    print(format_banner(f"{corpus_name} CORPUS ACQUISITION"))
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuration")
    print("=" * 80)
    print(f"Corpus path:        {corpus_path}")
    print(f"Text directory:     {text_dir}")
    print(f"Database path:      {db_path}")
    print(f"Text files found:   {len(file_year_pairs)}")
    print(f"Workers:            {workers}")
    print(f"Batch size:         {write_batch_size:,}")
    print(f"Overwrite DB:       {overwrite_db}")
    print()
    sys.stdout.flush()

    # Open database and create writer
    logger.info("Opening database...")
    with open_db(db_path, profile="write:packed24", create_if_missing=True) as db:
        writer = SentenceBatchWriter(db, batch_size=write_batch_size)

        total_sentences = 0
        total_errors = 0
        files_processed = 0

        print("Processing files...")
        print("=" * 80)
        sys.stdout.flush()

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

    # Print completion summary
    end_time = datetime.now()
    elapsed = end_time - start_time

    print()
    print(format_banner("ACQUISITION COMPLETE"))
    print(f"End Time:           {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:           {timedelta(seconds=int(elapsed.total_seconds()))}")
    print()
    print("Statistics")
    print("=" * 80)
    print(f"Files processed:    {files_processed}/{len(file_year_pairs)}")
    print(f"Sentences written:  {total_sentences:,}")
    print(f"Errors:             {total_errors}")
    print(f"Database path:      {db_path}")
    print()

    # Get database size
    try:
        with open_db(db_path, mode="r") as db:
            prop = db.get_property("rocksdb.total-sst-files-size")
            if prop:
                size_bytes = int(prop)
                print(f"Database size:      {format_bytes(size_bytes)}")
    except Exception:
        pass

    logger.info(f"Acquisition complete: {total_sentences:,} sentences written")
    print()
