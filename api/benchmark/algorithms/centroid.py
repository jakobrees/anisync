"""
Group Centroid

Pipeline:
  1. Per user: build a score-weighted embedding from their visible anime.
  2. Average the per-user embeddings into one group centroid vector.
  3. Issue a single pgvector query against that centroid, excluding all
     visible items across the group.
  4. Return results in retrieval order (similarity to centroid descending).

No per-user candidate pools, no re-ranking. One query, one ranked list.
This is the simplest possible group recommendation heuristic and serves
as a baseline.
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
    user_embeddings: list[np.ndarray] = []
    all_visible_ids: set[int] = set()

    for visible in visible_by_user.values():
        all_visible_ids.update(r.catalog_item_id for r in visible)
        user_embeddings.append(build_liked_query_embedding(db, visible))

    centroid = np.mean(user_embeddings, axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)

    candidates = retrieve_top_100(db, centroid, exclude_ids=all_visible_ids)
    return [item.id for item in candidates]
