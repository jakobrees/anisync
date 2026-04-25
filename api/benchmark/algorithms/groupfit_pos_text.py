"""
GroupFit Positive+Text

Simplified variant of GroupFit using only positive and text signals.
No negative centroid, one hyperparameter.

Formula:
  score(i) = (1 − α) · min_u max_j (e_i · liked_u_j)
           +      α  · mean_u (e_i · t_u)

α ∈ [0, 1] (read from BenchmarkConfig.groupfit_alpha):
  α = 0.0  →  pure liked-item fairness, text ignored
  α = 1.0  →  pure text alignment, liked items ignored

When cfg.use_msmarco=True, the text term uses the asymmetric msmarco model:
  - t_u is encoded with msmarco-MiniLM-L6-cos-v5
  - e_i in the text term uses item.embedding_msmarco (not item.embedding)
  - The positive term and pgvector retrieval always use item.embedding

Precompute hook: same pattern as groupfit — batch-embeds texts, pre-fetches
item embeddings, and pre-runs pgvector queries before the group loop.
Cache is reused across ablation runs that share the same base config.

Requires llm_translate.py for the text term (optional):
  cd api && python -m benchmark.llm_translate --visible-ratio <ratio>
"""
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session
from tqdm import tqdm

from app.embeddings import embed_texts, embed_texts_msmarco
from benchmark.config import BenchmarkConfig
from benchmark.methods.base import (
    UserRating,
    fetch_catalog_items_by_ids,
    retrieve_top_100,
    split_profile,
)

POSITIVE_THRESHOLD = 7

# ── Module-level precompute cache ─────────────────────────────────────────────

_cache_key: tuple | None = None
_text_cache: dict[str, np.ndarray | None] = {}
_liked_matrices: dict[str, np.ndarray] = {}
_retrieval_cache: dict[str, list] = {}
# msmarco embeddings for catalog items (populated when use_msmarco=True)
_item_msmarco_emb: dict[int, np.ndarray] = {}


def precompute(db: Session, groups: list, cfg: BenchmarkConfig) -> None:
    global _cache_key, _text_cache, _liked_matrices, _retrieval_cache, _item_msmarco_emb

    key = (cfg.profile_seed, cfg.visible_ratio, cfg.group_seed, cfg.num_groups, cfg.use_msmarco)
    if _cache_key == key:
        return

    cache_dir = Path(cfg.llm_cache_dir).resolve()
    all_profiles = {p.username: p for group in groups for p in group}

    # ── 1. Batch-embed all LLM texts ──────────────────────────────────────────
    texts_to_embed: list[str] = []
    usernames_to_embed: list[str] = []
    new_text_cache: dict[str, np.ndarray | None] = {}

    for username in all_profiles:
        path = cache_dir / f"{username}__{cfg.profile_seed}__{cfg.visible_ratio:.3f}.txt"
        if path.exists():
            texts_to_embed.append(path.read_text("utf-8").strip())
            usernames_to_embed.append(username)
        else:
            new_text_cache[username] = None

    if texts_to_embed:
        embed_fn = embed_texts_msmarco if cfg.use_msmarco else embed_texts
        embeddings = embed_fn(texts_to_embed, batch_size=128, show_progress_bar=True)
        for username, emb in zip(usernames_to_embed, embeddings):
            new_text_cache[username] = emb

    # ── 2. Pre-fetch all visible item embeddings ──────────────────────────────
    all_item_ids: set[int] = set()
    visible_splits: dict[str, list[UserRating]] = {}
    for username, profile in all_profiles.items():
        visible, _ = split_profile(profile, cfg.visible_ratio, cfg.profile_seed)
        visible_splits[username] = visible
        all_item_ids.update(r.catalog_item_id for r in visible)

    items = fetch_catalog_items_by_ids(db, list(all_item_ids))
    item_emb: dict[int, np.ndarray] = {
        iid: np.array(item.embedding, dtype=np.float32)
        for iid, item in items.items()
        if item.embedding is not None
    }

    # ── 3. Build per-user liked_matrix ────────────────────────────────────────
    new_liked: dict[str, np.ndarray] = {}
    for username, visible in visible_splits.items():
        liked = [r for r in visible if r.score >= POSITIVE_THRESHOLD] or visible
        liked_embs = [item_emb[r.catalog_item_id] for r in liked if r.catalog_item_id in item_emb]
        new_liked[username] = np.stack(liked_embs) if liked_embs else np.zeros((1, 384), dtype=np.float32)

    # ── 4. Pre-run all pgvector queries ───────────────────────────────────────
    new_retrieval: dict[str, list] = {}
    for username, visible in tqdm(visible_splits.items(), desc="pgvector queries", unit="user"):
        liked_mat = new_liked[username]
        liked_query = liked_mat.mean(axis=0)
        liked_query = liked_query / max(float(np.linalg.norm(liked_query)), 1e-12)
        visible_ids = {r.catalog_item_id for r in visible}
        new_retrieval[username] = retrieve_top_100(db, liked_query, exclude_ids=visible_ids)

    # ── 5. Pre-fetch msmarco embeddings for all candidate items ───────────────
    new_msmarco: dict[int, np.ndarray] = {}
    if cfg.use_msmarco:
        candidate_ids = {item.id for candidates in new_retrieval.values() for item in candidates}
        candidate_items_map = fetch_catalog_items_by_ids(db, list(candidate_ids))
        for iid, item in candidate_items_map.items():
            if item.embedding_msmarco is not None:
                new_msmarco[iid] = np.array(item.embedding_msmarco, dtype=np.float32)

    _text_cache = new_text_cache
    _liked_matrices = new_liked
    _retrieval_cache = new_retrieval
    _item_msmarco_emb = new_msmarco
    _cache_key = key
    model_tag = "msmarco" if cfg.use_msmarco else "standard"
    print(f"  [groupfit_pos_text/{model_tag}] precompute done: {len(all_profiles)} users, "
          f"{len(texts_to_embed)} texts embedded, {len(all_item_ids)} items fetched")


def recommend(
    db: Session,
    visible_by_user: dict[str, list[UserRating]],
    cfg: BenchmarkConfig,
) -> list[int]:
    alpha = cfg.groupfit_alpha

    candidate_pool: dict[int, object] = {}
    for username in visible_by_user:
        for item in _retrieval_cache.get(username, []):
            candidate_pool[item.id] = item

    if not candidate_pool:
        return []

    candidate_items = list(candidate_pool.values())
    c_matrix = np.stack(
        [np.array(item.embedding, dtype=np.float32) for item in candidate_items], axis=0
    )

    # For text term: use msmarco embeddings if available, else fall back to symmetric.
    if cfg.use_msmarco and _item_msmarco_emb:
        c_text_matrix = np.stack(
            [_item_msmarco_emb.get(item.id, np.array(item.embedding, dtype=np.float32))
             for item in candidate_items],
            axis=0,
        )
    else:
        c_text_matrix = c_matrix

    n = len(candidate_items)
    pos_rows: list[np.ndarray] = []
    text_rows: list[np.ndarray] = []

    for username in visible_by_user:
        liked_sims = _liked_matrices[username] @ c_matrix.T
        pos_rows.append(liked_sims.max(axis=0))

        t_u = _text_cache.get(username)
        text_rows.append(t_u @ c_text_matrix.T if t_u is not None else np.zeros(n, dtype=np.float32))

    pos_matrix = np.stack(pos_rows)
    text_matrix = np.stack(text_rows)

    scores = (1 - alpha) * pos_matrix.min(axis=0) + alpha * text_matrix.mean(axis=0)
    return [candidate_items[int(i)].id for i in np.argsort(-scores)]
