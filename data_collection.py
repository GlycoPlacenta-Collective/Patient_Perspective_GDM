"""
Step 2: Data Collection and Filtering
======================================
Loads the prefiltered archives, restricts them to the study period,
applies quality filters.

Output: filtered_posts.csv (one row per post or comment).

Data source: Pushshift / Arctic Shift (https://github.com/ArthurHeitmann/arctic_shift)

Period: January 2023– April 2026

"""

import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2026, 4, 30)
 
MIN_WORD_COUNT = 10
MIN_ALPHA_RATIO = 0.50

# ============================================================================
# Data Loading
# ============================================================================

def _load_processFiles_module():
    """Import the Arctic Shift processFiles helper."""
    arctic_shift_path = Path(__file__).parent / 'arctic_shift' / 'scripts' # requires arctic_shift repo to be cloned
    sys.path.insert(0, str(arctic_shift_path))
    import processFiles
    return processFiles
 
 
def load_single_file(processFiles, file_path, start_date, end_date):
    """
    Load one prefiltered archive file (one month of comments or submissions)
    and keep rows in the study date window.
 
    Returns: list of row dicts with keys user_id, subreddit, message, date, 
    and title as produced by processFiles.processFile.
    """
    processFiles.fileOrFolderPath = str(file_path)
    rows = processFiles.main()
 
    def keep(row):
        ts = row.get("date", 0)
        if start_date and ts < start_date.timestamp():
            return False
        if end_date and ts > end_date.timestamp():
            return False
        return True
 
    return [r for r in rows if keep(r)]
 
 
# ============================================================================
# Filtering
# ============================================================================
 
 
def filter_comments(df):
    """Apply quality filters to comments DataFrame."""
    filtered = df.copy()
 
    # Remove deleted/removed content
    filtered = filtered[~filtered["message"].isin(["[removed]", "[deleted]"])]
    filtered = filtered.dropna(subset=["message"])
 
    # Minimum word count
    filtered = filtered[
        filtered["message"].str.split().str.len() >= MIN_WORD_COUNT
    ]
 
    # Must contain spaces (not a single token)
    filtered = filtered[
        filtered["message"].str[:2048].str.contains(" ")
    ]
 
    # At least 50% alphabetic characters
    filtered = filtered[
        filtered["message"].apply(
            lambda x: sum(c.isalpha() for c in x) / max(len(x), 1) >= MIN_ALPHA_RATIO
        )
    ]
 
 
    return filtered
 
 
def filter_posts(df):
    """Apply quality filters to posts (submissions) DataFrame."""
    filtered = df.copy()
 
    # Remove deleted/removed content
    filtered = filtered[~filtered["message"].isin(["[removed]", "[deleted]"])]
    filtered = filtered.dropna(subset=["title"])
    filtered = filtered.dropna(subset=["message"])
 
    # Title must contain spaces
    filtered = filtered[
        filtered["title"].str[:2048].str.contains(" ")
    ]
 
    # Combine title and message for length/quality checks
    filtered["combined_title_message"] = filtered["title"] + " " + filtered["message"]
 
    # Minimum word count
    filtered = filtered[
        filtered["combined_title_message"].str.split().str.len() >= MIN_WORD_COUNT
    ]
 
    # At least 50% alphabetic characters
    filtered = filtered[
        filtered["combined_title_message"].apply(
            lambda x: sum(c.isalpha() for c in x) / max(len(x), 1) >= MIN_ALPHA_RATIO
        )
    ]

 
    return filtered
 
 
# ============================================================================
# Main
# ============================================================================
 
OUTPUT_PATH = Path("filtered_posts.csv")
LEDGER_PATH = Path("data_collection_processed_files.txt")  # lets an interrupted run resume
 
 
def already_processed():
    if not LEDGER_PATH.exists():
        return set()
    return set(LEDGER_PATH.read_text().splitlines())
 
 
def mark_processed(file_path):
    with open(LEDGER_PATH, "a") as f:
        f.write(f"{file_path}\n")
 
 
def process_one_file(processFiles, file_path, is_comments, start_date, end_date):
    rows = load_single_file(processFiles, file_path, start_date, end_date)
 
    if is_comments:
        cols = ['user_id', 'subreddit', 'message', 'date']
    else:
        cols = ['user_id', 'subreddit', 'title', 'message', 'date']
 
    df = pd.DataFrame(rows).reindex(columns=cols)
    if is_comments:
        df = filter_comments(df)
        return df[["user_id", "subreddit", "message", "date"]]
    else:
        df = filter_posts(df)
        return df[["user_id", "subreddit", "combined_title_message", "date"]].rename(
            columns={"combined_title_message": "message"}
        )
 
 
if __name__ == "__main__":
    processFiles = _load_processFiles_module()
    base = Path(__file__).parent / 'archives' / 'filtered'
 
    comment_files = sorted((base / 'comments').glob("*.jsonl"))
    submission_files = sorted((base / 'submissions').glob("*.jsonl"))
    all_files = [(f, True) for f in comment_files] + [(f, False) for f in submission_files]
 
    done = already_processed()
    write_header = not OUTPUT_PATH.exists()
    total_kept = 0
 
    for i, (file_path, is_comments) in enumerate(all_files, 1):
        key = str(file_path)
        kind = "comments" if is_comments else "submissions"
        if key in done:
            print(f"[{i}/{len(all_files)}] Skipping {file_path.name} ({kind}) — already processed")
            continue
 
        print(f"[{i}/{len(all_files)}] Processing {file_path.name} ({kind})...")
        df = process_one_file(processFiles, file_path, is_comments, START_DATE, END_DATE)
 
        df.to_csv(OUTPUT_PATH, mode="a", header=write_header, index=False)
        write_header = False
        mark_processed(key)
 
        total_kept += len(df)
        print(f"  -> kept {len(df):,} rows (running total: {total_kept:,})")
 
    print(f"Done. Filtered rows appended to {OUTPUT_PATH}")