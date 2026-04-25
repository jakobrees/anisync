"""
GroupMatch Clustered

Pipeline:
  1. Per user: build a score-weighted embedding from their visible anime.
  2. Per user: retrieve top-100 candidates from the catalog via pgvector.
  3. Union and deduplicate the candidate pools.
  4. Score all candidates by mean GroupMatch (same as groupmatch_raw).
  5. Cluster candidates with silhouette-guided k-means.
  6. Promote the top-N items per cluster to the front of the ranked list,
     ordered globally by GroupMatch score (cluster_diverse_rerank).
  7. Append all remaining candidates below, also by GroupMatch score.

The clustering only affects which items are promoted to the top — it
ensures thematic diversity in the top results without re-ordering by
cluster membership. The GroupMatch score drives all ordering decisions.
"""
import numpy as np
from sqlalchemy.orm import Session

from app.ml.kmeans import choose_k_and_cluster
from benchmark.config import BenchmarkConfig
from benchmark.methods.base import (
    UserRating,
    build_liked_query_embedding,
    cluster_diverse_rerank,
    retrieve_top_100,
)


def recommend(
    db: Session,
    visible_by_user: dict[str, list[UserRating]],
    cfg: BenchmarkConfig,
) -> list[int]:
    query_embeddings: list[np.ndarray] = []
    candidate_pool: dict[int, object] = {}

    for visible in visible_by_user.values():
        visible_ids = {r.catalog_item_id for r in visible}
        q_emb = build_liked_query_embedding(db, visible)
        query_embeddings.append(q_emb)
        for item in retrieve_top_100(db, q_emb, exclude_ids=visible_ids):
            candidate_pool[item.id] = item

    if not candidate_pool:
        return []

    candidate_items = list(candidate_pool.values())
    q_matrix = np.stack(query_embeddings, axis=0)
    c_matrix = np.stack(
        [np.array(item.embedding, dtype=np.float32) for item in candidate_items], axis=0
    )

    avg_scores = (q_matrix @ c_matrix.T).mean(axis=0)
    kmeans = choose_k_and_cluster(c_matrix)

    return cluster_diverse_rerank(
        candidate_items, avg_scores, kmeans.assignments, kmeans.k, n_per_cluster=5
    )
