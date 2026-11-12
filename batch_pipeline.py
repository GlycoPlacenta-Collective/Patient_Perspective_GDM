"""
Step 3: Batch Pipeline Orchestrator
============================
Batch pipeline wrapper around the OpenAI Batch API for large-scale
runs that exceed a single batch's limits (50,000 requests / 200 MB per
file) and/or your account's daily batch queue token budget (TPD).

Runs both LLM stages of the study. Prompts and schemas live in prompts.py:
  Stage 1 - self-use classification   (SELF_USE_SYSTEM_PROMPT / SELF_USE_SCHEMA)
  Stage 2 - narrative extraction      (GDM_EXPERIENCE_SYSTEM_PROMPT / GDM_EXPERIENCE_SCHEMA)

What it does
----------------
  1. Split into multiple batch files (OpenAI hard limit: 50,000 requests
     or 200 MB per file, whichever comes first)
  2. Avoid submitting more input tokens than your daily batch queue
     budget allows 
  3. Survive a crash/restart without redoing completed work 


Usage
-----
Requires OPENAI_API_KEY.
 
python batch_pipeline.py self_use     # Stage 1
python batch_pipeline.py narrative    # Stage 2
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
 
import pandas as pd
from openai import OpenAI
 
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")
 
    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:
    # Falls back to the standard ~4 chars/token approximation if tiktoken's
    # vocab files can't be downloaded
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)
 
 
# ============================================================================
# Configuration
# ============================================================================
 
MAX_REQUESTS_PER_BATCH_FILE = 50_000       # OpenAI hard limit
MAX_FILE_SIZE_BYTES = 190 * 1024 * 1024    # 190 MB, leaving headroom under the 200 MB cap
DEFAULT_POLL_INTERVAL_SECONDS = 60
DEFAULT_BACKOFF_SECONDS = 300              # wait time when the queue is full
 
 
@dataclass
class ChunkRecord:
    chunk_id: str
    row_indices: list        
    status: str = "pending"    # pending -> submitted -> completed | failed
    batch_id: Optional[str] = None
    est_input_tokens: int = 0
    output_file: Optional[str] = None
    error: Optional[str] = None
 
 
class Manifest:
    """
    Tracks chunk status on disk so a crash/restart doesn't redo completed
    work. One manifest per pipeline stage (self-use vs. narrative
    extraction).
    """
 
    def __init__(self, manifest_path: str):
        self.path = Path(manifest_path)
        self.chunks: dict[str, ChunkRecord] = {}
        if self.path.exists():
            self._load()
 
    def _load(self):
        with open(self.path) as f:
            data = json.load(f)
        for cid, rec in data.items():
            self.chunks[cid] = ChunkRecord(**rec)
 
    def save(self):
        data = {cid: rec.__dict__ for cid, rec in self.chunks.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)
 
    def pending_chunks(self):
        return [c for c in self.chunks.values() if c.status == "pending"]
 
    def in_flight_chunks(self):
        return [c for c in self.chunks.values() if c.status == "submitted"]
 
    def in_flight_tokens(self):
        return sum(c.est_input_tokens for c in self.in_flight_chunks())
 
    def all_done(self):
        return all(c.status in ("completed", "failed") for c in self.chunks.values())


# ============================================================================
# Chunking
# ============================================================================
 
 
def build_chunks(
    df: pd.DataFrame,
    system_prompt: str,
    build_user_message: Callable[[pd.Series], str],
    id_column: str,
    max_requests: int = MAX_REQUESTS_PER_BATCH_FILE,
    max_file_bytes: int = MAX_FILE_SIZE_BYTES,
) -> list[ChunkRecord]:
    """
    Split df into chunks that each respect the request-count and file-size
    caps, estimating serialized size as we go
    """
    sys_tokens = count_tokens(system_prompt)
    chunks = []
    current_indices = []
    current_bytes = 0
    current_tokens = 0
 
    for idx, row in df.iterrows():
        user_msg = build_user_message(row)
        # Rough serialized size of one JSONL line
        line_bytes = len(system_prompt.encode()) + len(user_msg.encode()) + 300
        line_tokens = sys_tokens + count_tokens(user_msg)
 
        would_exceed_count = len(current_indices) + 1 > max_requests
        would_exceed_size = current_bytes + line_bytes > max_file_bytes
 
        if current_indices and (would_exceed_count or would_exceed_size):
            chunks.append(_finalize_chunk(chunks, current_indices, current_tokens))
            current_indices, current_bytes, current_tokens = [], 0, 0
 
        current_indices.append(idx)
        current_bytes += line_bytes
        current_tokens += line_tokens
 
    if current_indices:
        chunks.append(_finalize_chunk(chunks, current_indices, current_tokens))
 
    return chunks
 
 
def _finalize_chunk(existing_chunks, indices, est_tokens) -> ChunkRecord:
    chunk_id = f"chunk-{len(existing_chunks):05d}"
    return ChunkRecord(chunk_id=chunk_id, row_indices=list(indices), est_input_tokens=est_tokens)
 
 
# ============================================================================
# Batch submission / polling / parsing
# ============================================================================
 
 
def submit_chunk(
    client: OpenAI,
    chunk: ChunkRecord,
    df: pd.DataFrame,
    system_prompt: str,
    schema: dict,
    schema_name: str,
    model: str,
    build_user_message: Callable[[pd.Series], str],
    id_column: str,
    max_tokens: int,
    work_dir: Path,
):
    """Write this chunk's input file, upload it, and submit the batch."""
    input_path = work_dir / f"{chunk.chunk_id}_input.jsonl"
    with open(input_path, "w") as f:
        for idx in chunk.row_indices:
            row = df.loc[idx]
            task = {
                "custom_id": f"task-{row[id_column]}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": build_user_message(row)},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                    },
                    "max_tokens": max_tokens,
                },
            }
            f.write(json.dumps(task) + "\n")
 
    batch_file = client.files.create(file=open(input_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    chunk.batch_id = batch.id
    chunk.status = "submitted"
    print(f"  Submitted {chunk.chunk_id} -> batch {batch.id} "
          f"({len(chunk.row_indices):,} requests, ~{chunk.est_input_tokens:,} input tokens)")
 
 
def poll_and_collect(client: OpenAI, chunk: ChunkRecord, work_dir: Path, results_path: Path):
    """
    Check one in-flight chunk's status. If completed, download + parse
    results, append to the master results CSV, and mark the chunk done.
    Returns True if the chunk finished (completed or failed), else False.
    """
    batch = client.batches.retrieve(chunk.batch_id)
    if batch.status not in ("completed", "failed", "expired", "cancelled"):
        return False
 
    if batch.status != "completed":
        chunk.status = "failed"
        chunk.error = f"batch ended with status={batch.status}"
        print(f"  {chunk.chunk_id} ({chunk.batch_id}) ended with status={batch.status}")
        return True
 
    output_path = work_dir / f"{chunk.chunk_id}_output.jsonl"
    content = client.files.content(batch.output_file_id)
    with open(output_path, "wb") as f:
        f.write(content.read())
 
    records = []
    with open(output_path) as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]
            try:
                body = result["response"]["body"]["choices"][0]["message"]["content"]
                parsed = json.loads(body)
                records.append({"custom_id": custom_id, **parsed})
            except (KeyError, IndexError, json.JSONDecodeError):
                records.append({"custom_id": custom_id, "_parse_error": True})
 
    result_df = pd.DataFrame(records)
    write_header = not results_path.exists()
    result_df.to_csv(results_path, mode="a", header=write_header, index=False)
 
    chunk.status = "completed"
    chunk.output_file = str(output_path)
    print(f"  {chunk.chunk_id} completed -> {len(records):,} results appended to {results_path.name}")
    return True
 
 
# ============================================================================
# Main driver
# ============================================================================
 
 
def run_pipeline_stage(
    df: pd.DataFrame,
    system_prompt: str,
    schema: dict,
    schema_name: str,
    model: str,
    build_user_message: Callable[[pd.Series], str],
    id_column: str,
    manifest_path: str,
    results_path: str,
    work_dir: str,
    daily_token_budget: int,
    max_tokens: int = 2048,
    poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
):
    """
    Drives one pipeline stage (self-use classification OR narrative
    extraction) through to completion. Splits the input df into chunks, 
    submits them to the OpenAI Batch API

    """
    client = OpenAI()  # uses OPENAI_API_KEY
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(results_path)
 
    manifest = Manifest(manifest_path)
    if not manifest.chunks:
        print(f"Building chunks for {len(df):,} rows...")
        chunks = build_chunks(df, system_prompt, build_user_message, id_column)
        for c in chunks:
            manifest.chunks[c.chunk_id] = c
        manifest.save()
        print(f"Created {len(chunks)} chunk(s).")
    else:
        print(f"Resuming from existing manifest: "
              f"{sum(1 for c in manifest.chunks.values() if c.status=='completed')} completed, "
              f"{sum(1 for c in manifest.chunks.values() if c.status=='submitted')} in flight, "
              f"{sum(1 for c in manifest.chunks.values() if c.status=='pending')} pending.")
 
    while not manifest.all_done():
        # 1. Check in-flight chunks for completion
        for chunk in manifest.in_flight_chunks():
            poll_and_collect(client, chunk, work_dir, results_path)
        manifest.save()
 
        # 2. Submit more work if there's budget for it
        in_flight_tokens = manifest.in_flight_tokens()
        for chunk in manifest.pending_chunks():
            if in_flight_tokens + chunk.est_input_tokens > daily_token_budget:
                break  # not enough headroom right now -- wait for something to finish
            try:
                submit_chunk(
                    client, chunk, df, system_prompt, schema, schema_name,
                    model, build_user_message, id_column, max_tokens, work_dir,
                )
                in_flight_tokens += chunk.est_input_tokens
                manifest.save()
            except Exception as e:
                # If the queue is full or the request fails, retry later
                print(f"  Could not submit {chunk.chunk_id} ({e}); will retry.")
                time.sleep(DEFAULT_BACKOFF_SECONDS)
                break
 
        if not manifest.all_done():
            print(f"  ...waiting ({poll_interval}s). "
                  f"In-flight tokens: ~{manifest.in_flight_tokens():,} / {daily_token_budget:,}")
            time.sleep(poll_interval)
 
    print(f"Stage complete. Results in {results_path}")
 
 
# ============================================================================
# Run Stage 1 or Stage 2
# ============================================================================
 
MODEL = "gpt-4.1-mini"
DAILY_TOKEN_BUDGET = 18_000_000  #Safety margin
 
 
def run_self_use():
    """Stage 1: classify every filtered post for personal GDM experience."""
    from prompts import SELF_USE_SYSTEM_PROMPT, SELF_USE_SCHEMA, build_user_message
 
    df = pd.read_csv("filtered_posts.csv").reset_index().rename(columns={"index": "_task_index"})
 
    run_pipeline_stage(
        df=df,
        system_prompt=SELF_USE_SYSTEM_PROMPT,
        schema=SELF_USE_SCHEMA,
        schema_name="self_use_classification",
        model=MODEL,
        build_user_message=build_user_message,
        id_column="_task_index",
        manifest_path="manifests/self_use_manifest.json",
        results_path="results/self_use_results.csv",
        work_dir="batch_work/self_use",
        daily_token_budget=DAILY_TOKEN_BUDGET,
        max_tokens=512,
    )

    #Full results
    results = pd.read_csv("results/self_use_results.csv")
    results["_task_index"] = results["custom_id"].str.replace("task-", "", regex=False).astype(int)
    merged = df.merge(results.drop(columns="custom_id"), on="_task_index", how="left")
    merged.to_csv("results/self_use_full.csv", index=False)
    
    #Keep only the posts where self_use==True for Stage 2
    self_use_df = merged[merged["self_use"] == True]
    self_use_df.to_csv("self_use_posts.csv", index=False)
    print(f"Self-use posts: {len(self_use_df):,} / {len(merged):,}")
 
 
def run_narrative():
    """Stage 2: extract pathway experiences from self-use posts."""
    from prompts import (
        GDM_EXPERIENCE_SYSTEM_PROMPT, GDM_EXPERIENCE_SCHEMA,
        build_user_message, flatten_experiences,
    )
 
    df = pd.read_csv("self_use_posts.csv").set_index("_task_index", drop=False)
    work_dir = "batch_work/narrative"
 
    run_pipeline_stage(
        df=df,
        system_prompt=GDM_EXPERIENCE_SYSTEM_PROMPT,
        schema=GDM_EXPERIENCE_SCHEMA,
        schema_name="gdm_experience_extraction",
        model=MODEL,
        build_user_message=build_user_message,
        id_column="_task_index",
        manifest_path="manifests/narrative_manifest.json",
        results_path="results/narrative_results.csv",
        work_dir=work_dir,
        daily_token_budget=DAILY_TOKEN_BUDGET,
        max_tokens=1000,
    )
 
    # One row per extracted experience
    experiences = flatten_experiences(sorted(Path(work_dir).glob("*_output.jsonl")))
    experiences["_task_index"] = (
        experiences["custom_id"].str.replace("task-", "", regex=False).astype(int)
    )
    merged = df.reset_index(drop=True).merge(
        experiences.drop(columns="custom_id"), on="_task_index", how="left"
    )
    merged.to_csv("posts_with_experiences.csv", index=False)
    print(f"Saved {len(merged):,} rows to posts_with_experiences.csv")
    print(experiences["domain"].value_counts().to_string())

 
 
if __name__ == "__main__":
    import sys
 
    stages = {"self_use": run_self_use, "narrative": run_narrative}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        sys.exit(f"Usage: python {Path(__file__).name} [{' | '.join(stages)}]")
    stages[sys.argv[1]]()