"""
GroupMatch Raw

Pipeline:
  1. Per user: build a score-weighted embedding from their visible anime.
  2. Per user: retrieve top-100 candidates from the catalog via pgvector.
  3. Union and deduplicate the candidate pools.
  4. Rank all candidates by mean GroupMatch score (average cosine similarity
     across all user query vectors). No clustering.

This is the AniSync production pipeline minus the clustering step.
"""
import numpy as np
from sqlalchemy.orm import Session

from benchmark.config import BenchmarkConfig
from benchmark.methods.base import UserRating, build_liked_query_embedding, retrieve_top_100


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
    return [candidate_items[int(i)].id for i in np.argsort(-avg_scores)]
