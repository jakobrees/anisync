from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import CatalogItem


@dataclass
class UserRating:
    mal_id: int
    catalog_item_id: int
    score: int


@dataclass
class UserProfile:
    username: str
    ratings: list[UserRating]


def split_profile(
    profile: UserProfile,
    visible_ratio: float,
    profile_seed: int,
) -> tuple[list[UserRating], list[UserRating]]:
    ratings = sorted(profile.ratings, key=lambda r: r.mal_id)
    n_visible = max(1, round(len(ratings) * visible_ratio))
    rng = np.random.default_rng(hash((profile.username, profile_seed)) & 0xFFFF_FFFF)
    perm = rng.permutation(len(ratings))
    visible_set = set(perm[:n_visible].tolist())
    visible = [ratings[i] for i in range(len(ratings)) if i in visible_set]
    hidden = [ratings[i] for i in range(len(ratings)) if i not in visible_set]
    return visible, hidden


def ndcg_at_k(ranked_ids: list[int], hidden_rel: dict[int, int], k: int) -> float:
    dcg = sum(
        hidden_rel.get(iid, 0) / math.log2(i + 2)
        for i, iid in enumerate(ranked_ids[:k])
    )
    ideal = sorted(hidden_rel.values(), reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def fetch_catalog_items_by_ids(db: Session, ids: list[int]) -> dict[int, CatalogItem]:
    if not ids:
        return {}
    items = db.scalars(select(CatalogItem).where(CatalogItem.id.in_(ids))).all()
    return {item.id: item for item in items}


def retrieve_top_100(
    db: Session,
    query_embedding: np.ndarray,
    *,
    exclude_ids: set[int] | None = None,
) -> list[CatalogItem]:
    distance_expr = CatalogItem.embedding.cosine_distance(
        query_embedding.astype(float).tolist()
    )
    stmt = select(CatalogItem).order_by(distance_expr.asc()).limit(200)
    if exclude_ids:
        stmt = stmt.where(~CatalogItem.id.in_(list(exclude_ids)))
    rows = db.scalars(stmt).all()
    return list(rows[:100])


def retrieve_top_100_msmarco(
    db: Session,
    query_embedding: np.ndarray,
    *,
    exclude_ids: set[int] | None = None,
) -> list[CatalogItem]:
    """pgvector retrieval using the embedding_msmarco column (text→item path)."""
    distance_expr = CatalogItem.embedding_msmarco.cosine_distance(
        query_embedding.astype(float).tolist()
    )
    stmt = (
        select(CatalogItem)
        .where(CatalogItem.embedding_msmarco.is_not(None))
        .order_by(distance_expr.asc())
        .limit(200)
    )
    if exclude_ids:
        stmt = stmt.where(~CatalogItem.id.in_(list(exclude_ids)))
    rows = db.scalars(stmt).all()
    return list(rows[:100])


def cluster_diverse_rerank(
    candidate_items: list,
    avg_scores: np.ndarray,
    assignments: np.ndarray,
    k: int,
    n_per_cluster: int = 5,
) -> list[int]:
    """Benchmark utility: promote top-N items per cluster, then fall back to GroupMatch order.

    This isolates the cluster-diversity effect from the GroupMatch scoring signal:
      1. From each cluster, select the top-N items by GroupMatch score.
      2. Sort those promoted items globally by GroupMatch score (best first).
      3. Append all remaining items below, also sorted by GroupMatch score.

    The result has diverse cluster representation at the top while preserving
    GroupMatch ordering everywhere. Nothing is dropped.
    """
    promoted_indices: set[int] = set()
    for c in range(k):
        members = np.where(assignments == c)[0]
        top_indices = members[np.argsort(-avg_scores[members])[:n_per_cluster]]
        promoted_indices.update(top_indices.tolist())

    order = np.argsort(-avg_scores)
    promoted = [int(i) for i in order if i in promoted_indices]
    rest = [int(i) for i in order if i not in promoted_indices]
    return [candidate_items[i].id for i in promoted + rest]


def build_proxy_relevant_set(
    db: Session,
    hidden: list[UserRating],
    n_top: int = 10,
    n_neighbors: int = 10,
) -> dict[int, float]:
    """Expand hidden ratings into a proxy relevance set via embedding neighbors.

    Motivation: user ratings are from 2018; the catalog contains anime up to 2026.
    A user who loved a show will likely also love similar newer shows that can't
    appear in their hidden set. This function bridges that gap.

    Algorithm:
      1. Take the top-n_top hidden items by score (score > 5 only — positive signal).
      2. For each, find its n_neighbors nearest catalog entries by embedding distance.
      3. Assign each proxy item a relevance of max(nominating_score - 5, 0).
         When multiple hidden items nominate the same proxy, take the max score.

    Returns: {catalog_item_id: relevance} — suitable for ndcg_at_k().
    """
    top_hidden = sorted(
        [r for r in hidden if r.score > 5],
        key=lambda r: -r.score,
    )[:n_top]

    if not top_hidden:
        return {}

    seed_items = fetch_catalog_items_by_ids(db, [r.catalog_item_id for r in top_hidden])

    proxy_rel: dict[int, float] = {}
    for r in top_hidden:
        item = seed_items.get(r.catalog_item_id)
        if item is None or item.embedding is None:
            continue
        rel = float(r.score - 5)
        emb = np.array(item.embedding, dtype=np.float32)
        distance_expr = CatalogItem.embedding.cosine_distance(emb.astype(float).tolist())
        stmt = select(CatalogItem.id).order_by(distance_expr.asc()).limit(n_neighbors)
        for nid in db.scalars(stmt).all():
            proxy_rel[nid] = max(proxy_rel.get(nid, 0.0), rel)

    return proxy_rel


def build_liked_query_embedding(db: Session, visible: list[UserRating], threshold: int = 7) -> np.ndarray:
    """Query embedding built from liked items only (score >= threshold).

    Falls back to all visible items if none meet the threshold.
    Items are averaged with equal weight (no score weighting).
    """
    liked = [r for r in visible if r.score >= threshold] or visible
    items = fetch_catalog_items_by_ids(db, [r.catalog_item_id for r in liked])
    embs = [
        np.array(items[r.catalog_item_id].embedding, dtype=np.float32)
        for r in liked
        if r.catalog_item_id in items and items[r.catalog_item_id].embedding is not None
    ]
    if not embs:
        raise ValueError(f"No embeddings found for liked items")
    avg = np.mean(embs, axis=0)
    return avg / max(float(np.linalg.norm(avg)), 1e-12)


def build_profile_query_embedding(db: Session, visible: list[UserRating]) -> np.ndarray:
    ids = [r.catalog_item_id for r in visible]
    items = fetch_catalog_items_by_ids(db, ids)

    embs: list[np.ndarray] = []
    weights: list[float] = []
    for r in visible:
        item = items.get(r.catalog_item_id)
        if item is None or item.embedding is None:
            continue
        embs.append(np.array(item.embedding, dtype=np.float32))
        weights.append(float(r.score))

    if not embs:
        raise ValueError(f"No embeddings found for visible items: {ids}")

    avg = np.average(embs, axis=0, weights=weights)
    norm = np.linalg.norm(avg)
    return avg / max(norm, 1e-12)
