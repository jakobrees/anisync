# Benchmark Algorithms

Each file in this directory is one self-contained recommendation algorithm.
The benchmark runner (`run.py`) loads them by name and calls `recommend()`.

---

## Evaluation methodology

NDCG@K is computed against a **proxy relevance set**, not the raw hidden ratings.

Because user ratings are from 2018 and the catalog extends to 2026, directly
checking whether a recommended item appears in a user's hidden set undervalues
recommendations of equivalent newer anime the user never had the chance to rate.

The proxy set is built per user from their hidden ratings by `build_proxy_relevant_set`
in `methods/base.py`:

1. Take the top-10 hidden items by score (score > 5 only).
2. For each, find its 10 nearest catalog entries by embedding distance.
3. Assign each proxy item `relevance = max(nominating_score − 5, 0)`. When
   multiple hidden items nominate the same proxy, take the max relevance.

This gives up to 100 proxy items per user that serve as ground truth. The
original hidden items appear naturally (an item is its own nearest neighbor),
so the metric is a strict superset of the naive direct-match approach.

---

## Interface contract

Every algorithm must expose a single function with this exact signature:

```python
def recommend(
    db: Session,                                   # live SQLAlchemy session
    visible_by_user: dict[str, list[UserRating]],  # each user's visible ratings
    cfg: BenchmarkConfig,                          # run config (visible_ratio, etc.)
) -> list[int]:                                    # catalog_item_ids, best first
```

`UserRating` has three fields: `mal_id`, `catalog_item_id`, `score` (1–10).

### The output must be a full ranking

The benchmark scores results with NDCG@K, which measures rank position — not
set membership. Returning only a short list (e.g. top-10) means every position
beyond that contributes zero to the score, which makes the algorithm look
artificially bad at large K and prevents fair comparison.

**Always return every candidate your algorithm considered, ordered from best to
worst.** The benchmark will evaluate whatever prefix it needs.

### What this means for clustering algorithms

Clustering is a tool for deciding what goes *at the top*, not for filtering
what gets returned. The right pattern is:

1. Score all candidates with your primary signal (e.g. GroupMatch).
2. Use clusters to *promote* the best representatives of each theme to the front.
3. Append all remaining candidates below, sorted by the primary signal.

The `cluster_diverse_rerank` helper in `methods/base.py` implements this: it
selects the top-N items per cluster, sorts those promoted items globally by
the primary score, then appends the rest — also by score. Nothing is dropped
and the score ordering is never broken within either section.

---

## Current algorithms

### `centroid.py` — Group Centroid

Computes a per-user profile embedding (score-weighted average of visible anime
embeddings), averages those into a single group centroid, then issues one
pgvector nearest-neighbour query against the centroid. Returns results in
retrieval order (cosine similarity to the centroid, descending).

- **Retrieval**: one query per group
- **Ranking signal**: cosine similarity to the group centroid
- **Strength**: simple and fast
- **Weakness**: a single vector cannot capture diverging tastes within the group;
  users with niche preferences may find nothing relevant in the result

### `groupmatch_raw.py` — GroupMatch Raw

Issues one pgvector query per user (using their individual profile embedding),
unions the per-user candidate pools, and ranks all candidates by their mean
cosine similarity across all user query vectors (GroupMatch score).

- **Retrieval**: one query per user, results unioned
- **Ranking signal**: mean GroupMatch score — how well an item satisfies the
  whole group on average
- **Strength**: per-user retrieval ensures niche preferences are represented in
  the candidate pool; GroupMatch gives each user equal weight in the final score
- **Weakness**: items that strongly appeal to one user but not others may rank
  poorly even if they would be a good recommendation for that user

### `groupmatch_clustered.py` — GroupMatch Clustered

Same retrieval and GroupMatch scoring as `groupmatch_raw`, with a
cluster-diversity post-processing step via `cluster_diverse_rerank`. The top-5
items from each thematic cluster are promoted to the front of the list, sorted
globally by GroupMatch score. All remaining candidates follow, also by
GroupMatch score. Nothing is filtered out.

- **Retrieval**: one query per user, results unioned
- **Ranking signal**: GroupMatch score throughout; clustering only determines
  which items are promoted to the top positions
- **Strength**: guarantees thematic variety at the top without breaking the
  GroupMatch ordering
- **Weakness**: adds k-means cost; benefit depends on whether the candidate
  pool actually contains distinct thematic clusters

### `groupfit.py` — GroupFit

Implements the scoring formula from the design critique document. Uses three
distinct per-user signals and three distinct aggregations:

```
groupfit(i) = min_u  max_j (e_i · liked_u_j)     # fairness: every user matches
            − λ    * mean_u (e_i · neg_u)          # shared dislike penalty
            + β    * mean_u (e_i · t_u)            # shared mood alignment (optional)
```

- **Positive signal**: max cosine similarity to any liked item (score ≥ 7) — treats
  each liked anime as an independent taste direction, never collapsing them into a
  centroid. Aggregated with `min` so a candidate must match *every* user.
- **Negative signal**: cosine similarity to the mean embedding of disliked items
  (score ≤ 4). Aggregated with `mean` — one person's dislikes don't veto the group.
- **Text signal**: similarity to the LLM-cached preference text embedding `t_u`.
  Aggregated with `mean`. Zero and silently omitted if no cache entry exists.

Retrieval is identical to `groupmatch_raw` (score-weighted profile embedding →
pgvector query per user). The novelty is entirely in the scoring step.

Constants `LAMBDA=0.3` and `BETA=0.5` are the only free parameters.

Optionally requires `llm_translate.py` for the text term (degrades gracefully without it).

- **Strength**: the `min` on the positive term directly enforces that no user is left
  behind; independent liked-item modes prevent semantic averaging from blurring taste
- **Weakness**: `min` can be dominated by a single outlier user; no per-user k-means
  on taste modes (each liked item is a mode, which may be noisy for users with many ratings)

### `groupmatch_raw_llm.py` — GroupMatch Raw (LLM query)

Mirrors the live AniSync web app: each user's visible ratings are summarised
into a natural-language preference description by Claude (via `llm_translate.py`),
that text is embedded, and the embedding drives both pgvector retrieval and
GroupMatch ranking. Everything after query construction is identical to
`groupmatch_raw`.

Requires `llm_translate.py` to have been run first:
```bash
cd api && python -m benchmark.llm_translate --visible-ratio <ratio>
```

- **Retrieval**: one query per user (LLM text embedding), results unioned
- **Ranking signal**: mean GroupMatch score across LLM-derived query embeddings
- **Strength**: directly evaluates the production pipeline end-to-end
- **Weakness**: requires API calls and a pre-built LLM cache; sensitive to
  prompt quality and the LLM's ability to summarise taste faithfully

---

## Adding a new algorithm

1. Create `api/benchmark/algorithms/<your_name>.py` with a `recommend` function
   matching the interface above.
2. Add `"<your_name>"` to the `ALGORITHMS` list in `api/benchmark/run.py`.
3. Run: `cd api && python -m benchmark.run --algorithm <your_name>`

Shared utilities (pgvector retrieval, profile embedding construction, NDCG
computation, the visible/hidden split) are in `api/benchmark/methods/base.py`.
Import freely — they are stable and tested.
