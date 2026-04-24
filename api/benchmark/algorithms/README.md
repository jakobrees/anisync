# Benchmark Algorithms

Each file in this directory is one self-contained recommendation algorithm.
The benchmark runner (`run.py`) loads them by name and calls `recommend()`.

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
