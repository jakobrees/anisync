# AniSync Ranking Benchmark

Offline evaluation harness for group anime recommendation algorithms.
Each algorithm is scored by NDCG@K against real MyAnimeList ratings held out as ground truth.

For the full design and algorithm descriptions see [`description.tex`](description.tex).

---

## Prerequisites

1. The catalog must already be preprocessed into Postgres (`preprocess_catalog.py`).
2. Download `animelists_cleaned.csv` from [Kaggle — MyAnimeList dataset](https://www.kaggle.com/datasets/azathoth42/myanimelist) and place it at `data/raw/animelists_cleaned.csv`.
3. Install dependencies (adds `anthropic` and `pyyaml`):
   ```bash
   uv pip install -r requirements.txt
   ```

---

## Quickstart

```bash
# 1. Build user profiles (one-time, ~1 min)
cd api && python -m benchmark.build_profiles --csv ../data/raw/animelists_cleaned.csv

# 2. Run an algorithm (no inference cost)
cd api && python -m benchmark.run --algorithm centroid --num-groups 400 --ndcg-k 200

# 3. Summarise results
cd api && python -m benchmark.summarize
```

---

## Step-by-step

### 1. Build profiles

Extracts a MAL ID → catalog item mapping from the DB, joins it against the
ratings CSV, filters to users with at least `min_ratings` joined ratings, and
samples `max_users` users deterministically.

```bash
cd api && python -m benchmark.build_profiles \
  --csv ../data/raw/animelists_cleaned.csv \
  --max-users 2000 \
  --min-ratings 15
```

Outputs:
- `data/processed/mal_to_catalog.json` — MAL integer ID → catalog_item_id
- `data/processed/user_profiles.jsonl` — one profile per line

Run once; re-run only if the catalog or CSV changes.

### 2. Run an algorithm

```bash
cd api && python -m benchmark.run --algorithm <name> [options]
```

Available algorithms:

| Algorithm | Description |
|---|---|
| `centroid` | Average all users' profile embeddings → single pgvector query |
| `groupmatch_raw` | Per-user retrieval → union pool → rank by mean GroupMatch score |
| `groupmatch_clustered` | Same as `groupmatch_raw` + promotes top-5 per cluster to the front, globally ordered by GroupMatch |
| `groupmatch_raw_llm` | Same as `groupmatch_raw` but uses LLM-generated preference text as the query (mirrors the live web app) |
| `groupfit` | Per-item positive scoring (max over liked items) with `min` fairness aggregation, negative centroid penalty, and optional LLM text alignment |

Key options (all override `config.yaml`):

| Flag | Default | Description |
|---|---|---|
| `--num-groups` | 50 | Number of synthetic groups to score |
| `--group-size` | 4 | Users per group |
| `--visible-ratio` | 0.3 | Fraction of each user's ratings shown to the algorithm |
| `--ndcg-k` | 10 | Depth at which NDCG is evaluated |
| `--profile-seed` | 123 | Controls the visible/hidden split |
| `--group-seed` | 42 | Controls group sampling |

Each run writes one JSON to `benchmark/results/`.

### 3. Summarise results

```bash
# All algorithms, all configs
cd api && python -m benchmark.summarize

# Ablation over visible_ratio for one algorithm
cd api && python -m benchmark.summarize --ablate visible_ratio --algorithm groupmatch_clustered

# Cross-algorithm comparison at a fixed config
cd api && python -m benchmark.summarize --compare-methods --visible-ratio 0.3 --group-size 4
```

---

## Evaluation methodology

NDCG@K is computed against a **proxy relevance set** rather than the raw hidden ratings.

User ratings are from 2018; the catalog contains anime up to 2026. A user who
loved a show should receive credit when the algorithm surfaces thematically
similar newer anime — even if that show didn't exist when they wrote their
ratings. The proxy expansion bridges this gap via the embedding space.

For each user, `build_proxy_relevant_set` in `api/benchmark/methods/base.py`:

1. Selects the top-10 hidden items by score (score > 5 only).
2. For each, queries the 10 nearest catalog entries by embedding distance.
3. Assigns `relevance = max(nominating_score − 5, 0)` to each proxy item,
   taking the max when multiple hidden items nominate the same entry.

This yields up to 100 proxy items per user as ground truth. The original hidden
items appear naturally (each item is its own nearest neighbor), so the metric
is a strict superset of direct-match scoring.

---

## Configuration

Default values live in `benchmark/config.yaml`. Any value can be overridden
per-run with the corresponding CLI flag (e.g. `--ndcg-k 50`).

```yaml
min_ratings: 15        # minimum joined ratings to include a user in the pool
max_users: 2000        # size of the user sampling pool
visible_ratio: 0.3     # fraction of ratings revealed to the algorithm
profile_seed: 123      # seeds the visible/hidden split (deterministic)
group_size: 4          # users per synthetic group
group_seed: 42         # seeds group sampling (deterministic)
num_groups: 50         # groups to evaluate per run
ndcg_k: 10             # NDCG cutoff depth
llm_model: claude-haiku-4-5-20251001
llm_cache_dir: ../benchmark/cache/llm
results_dir: ../benchmark/results
```

---

## Adding a new algorithm

1. Create `api/benchmark/algorithms/<your_name>.py` with a single function:
   ```python
   def recommend(
       db,                                        # SQLAlchemy session
       visible_by_user: dict[str, list[UserRating]],  # visible ratings per user
       cfg: BenchmarkConfig,
   ) -> list[int]:                                # ranked catalog_item_ids
       ...
   ```
2. Add `"<your_name>"` to the `ALGORITHMS` list in `api/benchmark/run.py`.
3. Run it: `python -m benchmark.run --algorithm <your_name>`.

Shared utilities (pgvector retrieval, profile embedding, NDCG) are in
`api/benchmark/methods/base.py`.

---

## Directory layout

```
benchmark/              # config, cache, results (not Python)
  config.yaml
  cache/
    llm/                # LLM translation cache (gitignored)
  results/              # one JSON per run (gitignored)
  description.tex       # full algorithm and design documentation
  README.md             # this file

api/benchmark/          # Python source
  algorithms/
    centroid.py
    groupmatch_raw.py
    groupmatch_clustered.py
    groupmatch_raw_llm.py
    groupfit.py
  methods/
    base.py             # shared utilities
  build_profiles.py
  llm_translate.py
  run.py
  summarize.py
  config.py
```

---

## LLM-assisted algorithms (optional)

Some algorithms may use an LLM to translate a user's visible ratings into a
natural-language preference description, which is then embedded and used as the
query vector. This requires:

1. `ANTHROPIC_API_KEY` set in your `.env` file.
2. Running the translation step before `run.py`:
   ```bash
   cd api && python -m benchmark.llm_translate --visible-ratio 0.3
   ```
   This submits a Claude Batch API job (~50% cheaper than real-time), polls
   until complete, and caches results to `benchmark/cache/llm/`. Subsequent
   runs with the same config read from cache at no cost.

   To resume an interrupted batch:
   ```bash
   cd api && python -m benchmark.llm_translate --batch-id <id> --visible-ratio 0.3
   ```

Estimated cost: ~$0.92 per 1,000 users per `visible_ratio` value (batch rate).

The `groupmatch_raw_llm` algorithm uses these cached texts as its query vectors,
directly replicating the production AniSync pipeline in the benchmark.
