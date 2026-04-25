# AniSync Benchmark — Findings and Design Rationale

*April 2026*

---

## 1. What the Benchmark Measures

The benchmark scores group recommendation algorithms against real MyAnimeList ratings using NDCG@200 as the evaluation metric. The setup:

- **User pool**: 2,000 users sampled from MAL rating data (ratings from ~2018), each with at least 15 ratings
- **Groups**: 400 synthetic groups of 4 users, sampled deterministically
- **Visible/hidden split**: 50% of each user's ratings are shown to the algorithm; the other 50% are held out as ground truth
- **Proxy relevance set**: rather than checking whether recommended anime appear literally in the hidden set, the benchmark expands each user's hidden set via embedding neighbors. For each user, the top-10 highest-scored hidden items are taken, the 10 nearest catalog entries to each are found by cosine distance, and each proxy item is assigned `relevance = max(score − 5, 0)` of the best hidden item that nominated it. This gives up to 100 proxy relevant items per user.
- **Score**: NDCG@200, mean and worst-user across all groups

The proxy expansion was added specifically to address the temporal gap: ratings are from 2018, the catalog extends to 2026. A user who loved a show in 2018 should receive credit when the algorithm surfaces a similar show released in 2022.

---

## 2. Benchmark Results

All runs: 400 groups, group size 4, visible ratio 0.5, ndcg_k 200.

### Algorithm comparison

| Algorithm | mean NDCG | std | mean worst-user NDCG |
|---|---|---|---|
| `groupmatch_clustered` | 0.1065 | 0.0276 | 0.0538 |
| `groupmatch_raw` | 0.1058 | 0.0272 | 0.0528 |
| `groupfit_pos_text` | 0.1024\* | 0.0286 | 0.0509 |
| `groupfit` | 0.1006 | 0.0273 | 0.0492 |
| `centroid` | 0.0798 | 0.0235 | 0.0342 |
| `groupmatch_raw_llm` | 0.0470† | 0.0170 | 0.0188 |

\* *Measured with pre-liked-only retrieval code; newer runs with liked-only retrieval give ~0.1006.*
† *Stale result — run before proxy relevance set was introduced. Needs re-run.*

### GroupFit ablation (λ × β, 400 groups)

```
groupfit_beta         0.0       0.1       0.2       0.3       0.5
-----------------------------------------------------------------
0.0                0.0974    0.0969    0.0964    0.0957    0.0941
0.2                0.0990    0.0988    0.0983    0.0975    0.0961
0.5                0.1000    0.0998    0.0995    0.0990    0.0981
0.8                0.1002    0.1000    0.0997    0.0993    0.0984
1.0                0.1001    0.0999    0.0996    0.0992    0.0983
```

λ = negative penalty weight. β = text alignment weight.

### GroupFit Positive+Text ablation (α, 400 groups)

| α | mean NDCG |
|---|---|
| 0.3 | 0.1009 (best) |
| 0.4 | 0.1008 |
| 0.5 | 0.1006 |
| 0.0 | 0.0979 |
| 1.0 | 0.0963 |

α = blend weight between min-positive (0) and text alignment (1).

---

## 3. What the Results Say

**Negative signal should be dropped.** The λ ablation is monotonically decreasing in every row — every increase in negative penalty hurts. The mean-of-disliked-items centroid is diffuse enough that penalising similarity to it also penalises decent recommendations. λ = 0 in production.

**Text alignment adds a real but modest signal.** Moving from β=0 to β=0.8 gives roughly +3% NDCG at λ=0. The optimal α in the simpler pos+text model is 0.30–0.40, suggesting text should be a light complement to the primary positive signal, not the dominant factor.

**GroupMatch outperforms GroupFit on the benchmark.** The raw and clustered GroupMatch variants score 5–6% above GroupFit's best configuration. Clustering adds a further small but consistent gain over raw GroupMatch.

**GroupFit's min_u fairness constraint does not improve worst-user NDCG.** Even on the metric designed to capture worst-user experience, GroupMatch wins (0.0528 vs 0.0492). The min aggregation depresses overall quality without a measurable fairness payoff on this benchmark.

---

## 4. Benchmark Limitations

These results should be interpreted carefully because the benchmark has structural properties that favour certain algorithms.

**The proxy relevance set is built the same way GroupMatch retrieves.** The evaluation metric expands the hidden set by finding nearest embedding neighbors of liked items. GroupMatch retrieval also finds nearest embedding neighbors of liked items. Any algorithm that thinks in embedding-neighborhood terms is directly rewarded by a metric built on those same neighborhoods. This is a coherent metric — it measures whether the algorithm surfaces anime that are semantically close to what the user likes — but it is not a neutral one.

**NDCG measures average satisfaction, not fairness.** GroupFit's min_u aggregation is a deliberate fairness constraint: a candidate only scores well if every user in the group matches it. This sacrifices average quality to protect the worst-placed user. NDCG, being an average over users, cannot capture this property. The benchmark cannot distinguish between "everyone gets something decent" and "three people get great recommendations and one gets nothing."

**Ratings are from 2018; the catalog extends to 2026.** The proxy expansion partially addresses this, but users' tastes as expressed in 2018 ratings may not reflect their current preferences, and the embedding-neighbor expansion may not fully capture thematic continuity across eight years of anime production.

**The benchmark does not measure exploration quality.** AniSync's stated goal is coherent exploration — surfacing anime that users would not have found themselves, in a way that makes sense as a group activity. NDCG on historical ratings measures recall of already-known taste, not the quality of discovery. An algorithm that reliably surfaces popular, well-rated anime will score well on this benchmark without doing anything interesting.

---

## 5. Reasoning from Results to Design

The benchmark is useful for two things: ruling out clearly broken approaches (centroid at 0.0798 is too weak; pure text retrieval needs improvement) and validating that candidate approaches are in the right ballpark. It is not the right tool for choosing between GroupMatch and GroupFit, because the metric's structural properties favour GroupMatch by construction.

The product case for GroupFit rests on what the benchmark cannot measure:

- **Fairness in the room**: in a real session, one person being left out is a product failure. The mean NDCG of the group does not capture this. Min_u directly encodes the constraint that every participant must have at least one taste direction matched.
- **Exploration rather than recall**: the positive signal in GroupFit (max similarity to any liked item) rewards genuine match to individual taste vectors rather than popularity-weighted average similarity. This is more appropriate for a discovery product than for a recommendation engine optimising watch-through rates.
- **The text signal is valuable for exploration**: a user typing "I want something slow-paced and melancholic" is expressing a mood the embedding of their watched anime cannot capture. β≈0.8 in GroupFit uses this signal; GroupMatch ignores it entirely.
- **The negative signal empirically hurts**: this is the one clean finding from the ablation. λ = 0 in the production algorithm.

---

## 6. Final Design

### Recommendation algorithm

Use GroupFit Positive+Text as the scoring function within the existing clustering architecture:

```
score(i) = (1 − α) · min_u max_j (e_i · liked_u_j)
         +      α  · mean_u (e_i · t_u)

α ≈ 0.30    (from ablation; text as a light complement)
```

- **Retrieval**: per-user pgvector query using the mean of liked-item embeddings (score ≥ 7), one query per user, candidate pools unioned.
- **Scoring**: GroupFit pos+text formula above.
- **Clustering**: keep silhouette-guided k-means on the candidate pool, promote top-N per cluster to the front of the ranked list globally re-ordered by the GroupFit score (`cluster_diverse_rerank`). Clustering handles diversity; GroupFit handles fairness and taste alignment.
- **Negative ratings**: dropped from the scoring formula. The empirical evidence is clear: the negative centroid adds noise.

### Embedding model

Two embedding columns in `catalog_items`, serving two different retrieval paths:

| Column | Model | Dimension | Used for |
|---|---|---|---|
| `embedding` (existing) | `all-MiniLM-L6-v2` | 384 | item ↔ item similarity (liked-item retrieval) |
| `embedding_msmarco` (new) | `msmarco-MiniLM-L6-cos-v5` | 384 | text → item similarity (user preference text retrieval) |

**Rationale**: `all-MiniLM-L6-v2` is a symmetric model trained on sentence similarity — well-suited for comparing item embeddings to each other. `msmarco-MiniLM-L6-cos-v5` is fine-tuned on MS MARCO query→passage pairs, making it better at matching a natural language query ("I enjoy slow-burn psychological drama") against a document (anime metadata). Both are 384-dimensional and 22M parameters, so no storage or latency increase.

The text alignment term in GroupFit (the β · mean_u(e_i · t_u) component) uses `embedding_msmarco` at runtime. The liked-item retrieval path continues to use `embedding`.

This design cannot be directly validated by the current benchmark, which uses a single embedding model for all comparisons. The justification is structural: asymmetric training data is better matched to the query→document nature of the text-to-anime retrieval task. Whether this produces a measurable NDCG improvement could be tested by re-running the benchmark after preprocessing with the new model.
