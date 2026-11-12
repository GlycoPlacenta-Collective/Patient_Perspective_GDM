# Patient Perspectives on Gestational Diabetes: Reddit NLP Pipeline
 
Code for [PAPER TITLE / CITATION]. 
 
This pipeline collects Reddit posts and comments about gestational diabetes mellitus (GDM), identifies posts where the author describes their own GDM, and uses an LLM to extract the patient care pathway experiences they report. Each experience is classified into one of five domains: screening, diagnosis, antepartum monitoring, birth, and postpartum.
 
## Pipeline overview
 
| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `prefilter.py` | Streams the raw Arctic Shift monthly archives once and keeps rows from the study subreddits that match the GDM term regex | `archives/filtered/`, `table_s1_counts.csv` |
| 2 | `data_collection.py` | Restricts to the study period and applies quality filters | `filtered_posts.csv` |
| 3a | `batch_pipeline.py self_use` | LLM Stage 1: classifies whether the author personally has or had GDM, and extracts contextual metadata | `results/self_use_full.csv`, `self_use_posts.csv` |
| 3b | `batch_pipeline.py narrative` | LLM Stage 2: extracts GDM pathway experiences from self-use posts and assigns each to a domain | `posts_with_experiences.csv` (one row per experience) |
 
Supporting files:
- `config.py`: the list of study subreddits, shared by Steps 1 and 2
- `prompts.py`: system prompts, structured-output JSON schemas, the user-message format, and the Stage 2 flattening helper
## Data
 
Reddit data were obtained from the [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) archives. **No Reddit data are included in this repository.** The data include usernames and full post text, so they are not redistributed here.
 
**Subreddits :** BabyBumps, BabyBumpsandBeyondAu, BabyBumpsCanada, breakingmom, GestationalDiabetes, KetoGD, NewParents, PCOSandPregnant, PlusSizePregnancy, preeclampsia, PregnancyUK, pregnant, pregnantover35
 
**Study period:** 1 January 2023 through 29 April 2026.
 
**Inclusion criteria:**
- **GDM term match** (`prefilter.py`, case-insensitive regex): `gestational.?diabet|GDM|GD\b|pregnancy diabetes`. For submissions this is applied to the title and body together; for comments, to the body.
- **Quality filters** (`data_collection.py`): excludes `[removed]` and `[deleted]` content; requires at least 10 words, at least one space, and at least 50% alphabetic characters. Submissions also require a title containing a space.
## Requirements
 
- Python 3.10+
- An OpenAI API key with Batch API access
```bash
pip install pandas orjson zstandard openai tiktoken
```
 
`tiktoken` is optional and is used only to estimate token counts. Without it, the pipeline falls back to an estimate of about 4 characters per token.
 
## Setup
 
1. **Clone Arctic Shift** into `arctic_shift/` in the repo root:
```bash
   git clone https://github.com/ArthurHeitmann/arctic_shift.git
```
   The scripts use `arctic_shift/scripts/fileStreams.py`, `utils.py` and `processFiles.py`.
 
2. **Edit `arctic_shift/scripts/processFiles.py`.** Arctic Shift provides this file as a template to adapt. `data_collection.py` expects it to be edited so that:
   - `main(filter_fn=None)` and `processFile(path, filter_fn=None)` accept an optional row filter and **return** a list of dicts
   - each row is mapped to `user_id` (author), `subreddit`, `date` (created_utc), `score`, `id`, `is_comment`, and `message` (body for comments, selftext for submissions), plus `title` and `url` for submissions and `parent_id` and `link_id` for comments
3. **Download the archives** for the study period. Place the monthly `.zst` dumps in:
```
   archives/post_covid/comments/
   archives/post_covid/submissions/
```
   To use different folders, change the paths at the bottom of `prefilter.py`.
 
4. **Set your API key:**
```bash
   export OPENAI_API_KEY="sk-..."
```
 
## Usage
 
Run the steps in order:
 
```bash
python prefilter.py
python data_collection.py
python batch_pipeline.py self_use
python batch_pipeline.py narrative
```
 
Each step can resume after an interruption:
- `prefilter.py` skips archives whose output file already exists. Those skipped files are not counted, so run it once from scratch to get complete counts in `table_s1_counts.csv`.
- `data_collection.py` records finished files in `data_collection_processed_files.txt` and adds to `filtered_posts.csv`. To rerun from scratch, delete both files first.
- `batch_pipeline.py` tracks each batch in `manifests/`. Rerunning a stage resumes from where it stopped.
## LLM stages
 
Both stages use `gpt-4.1-mini` through the OpenAI Batch API (`/v1/chat/completions`) with strict JSON-schema structured output. The prompts and schemas are in `prompts.py`. Each post is sent as:
 
```
posted in r/<subreddit>
posted on <date>
<message>
```
 
**Stage 1: self-use classification** returns `self_use`, `confidence`, `management_methods`, `gravidity`, `gdm_recurrence`, `other_conditions`, `pregnancy_number`, `pre_existing_diabetes`, `pre_diabetic_history` and `reasoning`. Posts from authors with pre-existing type 1 or type 2 diabetes, and posts from authors still awaiting a test result, are classified as not self-use. Only posts with `self_use == True` go on to Stage 2.
 
**Stage 2: narrative extraction** returns a list of experiences, each with a normalized `experience` description, a `domain` (SCREENING, DIAGNOSIS, ANTEPARTUM_MONITORING, BIRTH or POSTPARTUM) and an `original_text` excerpt, plus a `no_pathway_experiences` flag. `flatten_experiences()` turns this into one row per experience. In the output, `no_pathway_experiences` means:
- `True`: the post had no extractable experiences
- `False`: the row is an extracted experience
- `None`: the model's response could not be parsed
**Batch handling.** `batch_pipeline.py` splits the input into batch files under OpenAI's limits (50,000 requests or 200 MB per file). It also keeps the estimated input tokens in the queue under `DAILY_TOKEN_BUDGET` (18M by default), and it backs off and retries if the queue is full. Adjust `DAILY_TOKEN_BUDGET` to your account's batch queue limit.
 
## Ethics
 
Only publicly available Reddit content was used. No Reddit data are redistributed. In the paper, content is paraphrased, with no direct quotes or identification of users. The study meets criteria for non-human subjects research under 45 CFR 46.102 and did not require Institutional Review Board review. 
 
 
## License
 
MIT License

Copyright (c) [2026] [Molly Campbell]
 