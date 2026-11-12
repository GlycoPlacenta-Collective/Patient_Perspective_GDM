"""
Step 1: Prefilter raw Arctic Shift archives
============================================
For memory efficiency, streams the raw monthly .zst archives once and keeps only rows that are
  (a) in one of the study subreddits (config.SUBREDDITS), AND
  (b) match the GDM term regex (TERMS_PATTERN, case-insensitive).
 
Output: .jsonl files

Input: *.zst, the Arctic Shift monthly archived Reddit data

Also writes table_s1_counts.csv: per-subreddit counts of rows retrieved
(subreddit match only) and rows term-matched - for reporting purposes.

"""

import csv
import sys
from collections import Counter
from pathlib import Path
import re
 
import orjson as json
 
sys.path.insert(0, str(Path(__file__).parent / 'arctic_shift' / 'scripts')) # requires arctic_shift repo to be cloned
from fileStreams import getFileJsonStream
 
from config import SUBREDDITS
 
# TERMS_PATTERN matches "gestational diabetes" and common abbreviations/misspellings 
TERMS_PATTERN = re.compile(
    r"gestational.?diabet|GDM|GD\b|pregnancy diabetes",
    re.IGNORECASE
)
 
retrieved_counts = Counter()
matched_counts = Counter()
skipped_files = []
 
 
def prefilter_file(input_path: Path, output_path: Path):
    if output_path.exists():
        print(f"Skipping {input_path.name} — already done")
        skipped_files.append(input_path.name)
        return
 
    print(f"Processing {input_path.name}...")
    kept = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        stream = getFileJsonStream(str(input_path), f_in)
        if stream is None:
            print(f"Skipping unknown format: {input_path.name}")
            return
        for row in stream:
            sub = row.get("subreddit")
            if sub not in SUBREDDITS:
                continue
            is_comment = "link_id" in row
            key = (sub, "comments" if is_comment else "submissions")
            retrieved_counts[key] += 1
            text = row.get("body") if is_comment else f"{row.get('title','')} {row.get('selftext','')}"
 
            if TERMS_PATTERN.search(text or ""):
                f_out.write(json.dumps(row) + b"\n")
                matched_counts[key] += 1
                kept += 1
 
    print(f"  → kept {kept:,} rows from {input_path.name}")
 
 
def prefilter_folder(input_folder: Path, output_folder: Path):
    files = sorted(input_folder.glob("*.zst"))
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] ", end="")
        out = output_folder / f.with_suffix(".jsonl").name
        prefilter_file(f, out)
 
 
if __name__ == "__main__":
    base = Path(__file__).parent / "archives"
 
    prefilter_folder(base / "post_covid" / "comments",    base / "filtered" / "comments") # Change to your input/output folders
    prefilter_folder(base / "post_covid" / "submissions", base / "filtered" / "submissions") # Change to your input/output folders
 
    out_csv = Path(__file__).parent / "table_s1_counts.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subreddit",
            "submissions_retrieved", "comments_retrieved", "total_retrieved",
            "submissions_term_matched", "comments_term_matched", "total_term_matched",
        ])
        for sub in sorted(SUBREDDITS):
            rs = retrieved_counts.get((sub, "submissions"), 0)
            rc = retrieved_counts.get((sub, "comments"), 0)
            ms = matched_counts.get((sub, "submissions"), 0)
            mc = matched_counts.get((sub, "comments"), 0)
            writer.writerow([sub, rs, rc, rs + rc, ms, mc, ms + mc])
            print(f"{sub}: retrieved {rs + rc:,}, term-matched {ms + mc:,}")
 
    if skipped_files:
        print(f"\nWARNING: {len(skipped_files)} archive(s) were skipped because their output "
              f"already existed, so {out_csv.name} is incomplete. Delete archives/filtered/ "
              f"and rerun for full counts.")
 
    print("Done — filtered archives saved to archives/filtered/")