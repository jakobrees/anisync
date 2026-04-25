"""
GroupMatch Raw LLM

Same pipeline as groupmatch_raw but uses the LLM-generated preference text
as each user's query instead of a score-weighted embedding average.

This mirrors the actual AniSync web app: a user types a preference
description, it gets embedded, and that embedding drives both retrieval
and GroupMatch ranking.

Requires llm_translate.py to have been run first:
  cd api && python -m benchmark.llm_translate --visible-ratio 0.5
"""
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.embeddings import embed_texts
from benchmark.config import BenchmarkConfig
from benchmark.methods.base import UserRating, retrieve_top_100


def _load_text(cache_dir: Path, username: str, profile_seed: int, visible_ratio: float) -> str:
    key = f"{username}__{profile_seed}__{visible_ratio:.3f}"
    path = cache_dir / f"{key}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"LLM cache missing for '{username}'. "
            f"Run: python -m benchmark.llm_translate --visible-ratio {visible_ratio}"
        )
    return path.read_text(encoding="utf-8").strip()


def recommend(
    db: Session,
    visible_by_user: dict[str, list[UserRating]],
    cfg: BenchmarkConfig,
) -> list[int]:
    cache_dir = Path(cfg.llm_cache_dir).resolve()

    query_embeddings: list[np.ndarray] = []
    candidate_pool: dict[int, object] = {}

    for username, visible in visible_by_user.items():
        visible_ids = {r.catalog_item_id for r in visible}
        text = _load_text(cache_dir, username, cfg.profile_seed, cfg.visible_ratio)
        q_emb = embed_texts([text])[0]
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
