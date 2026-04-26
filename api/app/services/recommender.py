import logging
from collections import Counter, defaultdict

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.embeddings import embed_texts
from app.ml.kmeans import choose_k_and_cluster
from app.models import CatalogItem, Room, RoomPreferenceSubmission


logger = logging.getLogger(__name__)


# GroupFit pos+text hyperparameters chosen by the offline benchmark
# (see benchmark/writeup.tex, Section "From benchmark to product"):
#   alpha = 0.30  → text alignment as a light complement to the positive term.
#   lambda = 0    → negative-similarity penalty is not used (proven harmful).
GROUPFIT_ALPHA = 0.30
POSITIVE_THRESHOLD = 7  # used in benchmark for filtering liked items by score
EMBEDDING_DIM = 384
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _safe_embedding(item: CatalogItem) -> np.ndarray | None:
    """
    Convert a CatalogItem.embedding to a (EMBEDDING_DIM,) float32 array.

    Returns None if the embedding is missing, malformed, or has the wrong
    dimensionality. The recommendation pipeline must filter these out before
    np.stack(), otherwise the entire compute crashes mid-room.
    """
    raw = getattr(item, "embedding", None)
    if raw is None:
        return None
    try:
        vector = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if vector.ndim != 1 or vector.shape[0] != EMBEDDING_DIM:
        return None
    if not np.isfinite(vector).all():
        return None
    return vector


def item_tags(item: CatalogItem) -> list[str]:
    """Return anime tags safely, with metadata_json fallback for older rows."""
    tags_json = getattr(item, "tags_json", None)
    if isinstance(tags_json, list):
        return [str(tag) for tag in tags_json]

    metadata = getattr(item, "metadata_json", None) or {}
    metadata_tags = metadata.get("tags", [])
    if isinstance(metadata_tags, list):
        return [str(tag) for tag in metadata_tags]

    return []


def public_item_payload(item: CatalogItem, group_match_score: float | None = None) -> dict:
    """
    Safe public item payload.

    `group_match_score` retains its name in the API for frontend compatibility,
    but its value is now the GroupFit pos+text score.
    """
    tags = item_tags(item)
    payload = {
        "catalog_item_id": item.id,
        "title": item.title,
        "media_type": item.media_type,
        "year": item.year,
        "status": item.status,
        "tags": tags[:6],
        "thumbnail_local_path": item.thumbnail_local_path,
        "image_local_path": item.image_local_path,
        "score": item.score,
    }

    if group_match_score is not None:
        payload["group_match_score"] = round(float(group_match_score), 4)

    return payload


def cluster_label(items: list[CatalogItem], scores_by_id: dict[int, float]) -> str:
    """Derive a short tag-based cluster label, weighted by GroupFit score."""
    weights: defaultdict[str, float] = defaultdict(float)

    for item in items:
        for tag in item_tags(item):
            weights[str(tag).lower()] += max(scores_by_id.get(item.id, 0.0), 0.0)

    if not weights:
        return ""

    top_tags = sorted(weights.items(), key=lambda pair: (-pair[1], pair[0]))[:2]
    return " / ".join(tag for tag, _ in top_tags)


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Cosine-normalize a vector, with a small floor to avoid divide-by-zero."""
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def _build_user_signals(
    submissions: list[RoomPreferenceSubmission],
    db: Session,
) -> tuple[
    dict[int, np.ndarray],   # liked_matrix_by_user_id  (n_liked_items × 384)
    dict[int, np.ndarray],   # text_vec_by_user_id      (384,)
    dict[int, np.ndarray],   # query_vec_by_user_id     (384,) — used for retrieval
]:
    """
    For each submission, compute:
      - a liked-item embedding matrix (rows = liked items)
      - a text query embedding (or None if no text)
      - a single retrieval query vector (mean of liked items, or text fallback)

    A user with no liked items has the text embedding repeated as their single
    "liked vector" so the GroupFit pos formula degenerates cleanly to a
    text-only score for them.
    """
    # 1. Embed all texts in one batch.
    texts_by_uid = {s.user_id: s.query_text.strip() for s in submissions}
    uids_with_text = [uid for uid, t in texts_by_uid.items() if t]

    text_vec_by_uid: dict[int, np.ndarray] = {}
    if uids_with_text:
        text_embs = embed_texts([texts_by_uid[uid] for uid in uids_with_text])
        for uid, emb in zip(uids_with_text, text_embs, strict=True):
            text_vec_by_uid[uid] = emb.astype(np.float32)

    # 2. Fetch all liked items in one query.
    all_liked_ids: set[int] = set()
    for submission in submissions:
        all_liked_ids.update(submission.liked_catalog_item_ids or [])

    liked_emb_by_id: dict[int, np.ndarray] = {}
    if all_liked_ids:
        rows = db.scalars(
            select(CatalogItem).where(CatalogItem.id.in_(list(all_liked_ids)))
        ).all()
        for item in rows:
            vector = _safe_embedding(item)
            if vector is not None:
                liked_emb_by_id[item.id] = vector

    # 3. Per-user matrices and queries.
    liked_matrix_by_uid: dict[int, np.ndarray] = {}
    query_vec_by_uid: dict[int, np.ndarray] = {}

    for submission in submissions:
        uid = submission.user_id
        liked_ids = [iid for iid in (submission.liked_catalog_item_ids or []) if iid in liked_emb_by_id]
        liked_embs = [liked_emb_by_id[iid] for iid in liked_ids]

        text_vec = text_vec_by_uid.get(uid)

        if liked_embs:
            liked_matrix = np.stack(liked_embs)
            query_vec = _normalize(liked_matrix.mean(axis=0))
        elif text_vec is not None:
            # Fallback: text vector serves as the user's single liked vector
            # AND as the retrieval query.
            liked_matrix = text_vec.reshape(1, -1)
            query_vec = text_vec
        else:
            # Should be unreachable: validated upstream that submission has
            # at least text or liked items.
            continue

        liked_matrix_by_uid[uid] = liked_matrix
        query_vec_by_uid[uid] = query_vec

    return liked_matrix_by_uid, text_vec_by_uid, query_vec_by_uid


def compute_recommendations(db: Session, room: Room) -> dict:
    """
    Public entrypoint for the GroupFit pos+text recommendation pipeline.

    Wraps `_compute_recommendations_inner` so that unexpected exceptions
    (DB blips, embedding model errors, numpy errors) are logged with the
    full traceback and re-raised as a generic RuntimeError. The
    user-facing 400 path is preserved through ValueError pass-through.
    """
    try:
        return _compute_recommendations_inner(db, room)
    except ValueError:
        # Validation problems (e.g. too few submissions) are surfaced to
        # the user as a 400 by main.compute_room.
        raise
    except Exception:
        logger.exception(
            "compute_recommendations: unexpected error in room id=%s code=%s",
            getattr(room, "id", None),
            getattr(room, "code", None),
        )
        raise RuntimeError(
            "Recommendation compute failed unexpectedly. Please retry; "
            "if it persists, contact the host."
        )


def _compute_recommendations_inner(db: Session, room: Room) -> dict:
    """
    GroupFit pos+text recommendation pipeline.

    score(i) = (1 - α) · min_u max_j (e_i · liked_u_j)
             +      α  · mean_u (e_i · t_u)

    α = 0.30  (light text complement to liked-item fairness)

    Steps:
      1. collect submissions (must have at least one of: text, liked items)
      2. embed texts; fetch liked-item embeddings
      3. per-user retrieval: pgvector cosine search against the user's mean
         liked-item embedding (or text embedding if they picked no items),
         within host hard constraints
      4. union/dedupe candidate pool
      5. score each candidate via GroupFit pos+text
      6. silhouette-guided k-means; cluster-diverse rerank
    """
    submissions = list(
        db.scalars(
            select(RoomPreferenceSubmission)
            .where(RoomPreferenceSubmission.room_id == room.id)
            .order_by(RoomPreferenceSubmission.user_id)
        )
    )

    submissions = [
        s for s in submissions
        if s.query_text.strip() or (s.liked_catalog_item_ids or [])
    ]

    if len(submissions) < 2:
        raise ValueError("At least 2 participants must submit preferences before recommendations can be generated.")

    allowed_types = room.hard_constraint_allowed_types_json or ["TV"]

    eligible_count = db.scalar(
        select(func.count(CatalogItem.id)).where(
            CatalogItem.year >= room.hard_constraint_year_start,
            CatalogItem.year <= room.hard_constraint_year_end,
            CatalogItem.media_type.in_(allowed_types),
        )
    )

    if not eligible_count:
        raise ValueError("The host's hard constraints leave no anime available for this room.")

    liked_matrix_by_uid, text_vec_by_uid, query_vec_by_uid = _build_user_signals(submissions, db)

    if len(query_vec_by_uid) < 2:
        raise ValueError("Could not derive enough preference signal from submissions.")

    # ── Per-user retrieval ────────────────────────────────────────────────────
    candidate_ids_by_user_id: dict[str, list[int]] = {}
    candidate_items_by_id: dict[int, CatalogItem] = {}

    # Liked items submitted by this room are excluded from candidates so we
    # never recommend back to a user something they already told us they like.
    excluded_ids: set[int] = set()
    for submission in submissions:
        excluded_ids.update(submission.liked_catalog_item_ids or [])

    for submission in submissions:
        uid = submission.user_id
        query_vec = query_vec_by_uid.get(uid)
        if query_vec is None:
            continue

        distance_expr = CatalogItem.embedding.cosine_distance(query_vec.astype(float).tolist())

        stmt = (
            select(CatalogItem)
            .where(
                CatalogItem.year >= room.hard_constraint_year_start,
                CatalogItem.year <= room.hard_constraint_year_end,
                CatalogItem.media_type.in_(allowed_types),
            )
            .order_by(distance_expr.asc())
            .limit(100 + len(excluded_ids))
        )

        if excluded_ids:
            stmt = stmt.where(~CatalogItem.id.in_(list(excluded_ids)))

        items = list(db.scalars(stmt))[:100]

        item_ids: list[int] = []
        for item in items:
            # Skip items with missing or malformed embeddings: pgvector cosine
            # search would not have returned them in normal operation, but
            # legacy rows or partial backfills can leak through.
            if _safe_embedding(item) is None:
                logger.warning(
                    "compute_recommendations: skipping catalog item %s with "
                    "missing or malformed embedding", item.id
                )
                continue
            candidate_items_by_id[item.id] = item
            item_ids.append(item.id)

        candidate_ids_by_user_id[str(uid)] = item_ids

    candidate_items = list(candidate_items_by_id.values())

    if len(candidate_items) < 2:
        raise ValueError("The combined candidate pool is too small to produce recommendations.")

    candidate_embeddings = np.stack(
        [_safe_embedding(item) for item in candidate_items],
        axis=0,
    )

    # ── GroupFit pos+text scoring ─────────────────────────────────────────────
    user_ids_in_score = list(query_vec_by_uid.keys())

    pos_rows: list[np.ndarray] = []
    text_rows: list[np.ndarray] = []

    for uid in user_ids_in_score:
        liked_sims = liked_matrix_by_uid[uid] @ candidate_embeddings.T
        pos_rows.append(liked_sims.max(axis=0))

        text_vec = text_vec_by_uid.get(uid)
        if text_vec is not None:
            text_rows.append(text_vec @ candidate_embeddings.T)
        else:
            # User gave no text — reuse the mean liked vector for the text term
            # so the formula stays well-defined without privileging this user.
            text_rows.append(query_vec_by_uid[uid] @ candidate_embeddings.T)

    pos_matrix = np.stack(pos_rows)    # (n_users, n_candidates)
    text_matrix = np.stack(text_rows)  # (n_users, n_candidates)

    scores = (1.0 - GROUPFIT_ALPHA) * pos_matrix.min(axis=0) + GROUPFIT_ALPHA * text_matrix.mean(axis=0)

    scores_by_item_id = {
        item.id: float(score)
        for item, score in zip(candidate_items, scores, strict=True)
    }

    # ── Clustering and cluster-diverse rerank ─────────────────────────────────
    kmeans_result = choose_k_and_cluster(candidate_embeddings)

    clusters_raw: list[dict] = []
    for cluster_index in range(kmeans_result.k):
        member_indexes = np.where(kmeans_result.assignments == cluster_index)[0].tolist()
        cluster_items = [candidate_items[i] for i in member_indexes]

        cluster_items.sort(key=lambda item: (-scores_by_item_id[item.id], item.title.lower()))
        top_items = cluster_items[:5]
        top_score = scores_by_item_id[top_items[0].id] if top_items else -999.0

        label = cluster_label(cluster_items, scores_by_item_id)

        clusters_raw.append(
            {
                "cluster_index": int(cluster_index),
                "cluster_label": label,
                "cluster_score": round(float(top_score), 4),
                "all_catalog_item_ids": [item.id for item in cluster_items],
                "top_items": [
                    public_item_payload(item, scores_by_item_id[item.id])
                    for item in top_items
                ],
                "top_two_items": [
                    public_item_payload(item, scores_by_item_id[item.id])
                    for item in cluster_items[:2]
                ],
            }
        )

    clusters_raw.sort(key=lambda cluster: (-cluster["cluster_score"], cluster["cluster_index"]))

    final_recommendations_by_id: dict[int, dict] = {}
    for cluster in clusters_raw:
        for item_payload in cluster["top_two_items"]:
            final_recommendations_by_id[item_payload["catalog_item_id"]] = item_payload

    final_recommendations = sorted(
        final_recommendations_by_id.values(),
        key=lambda item: (-item["group_match_score"], item["title"].lower()),
    )

    # ── Results envelope (private fields stripped before client send) ─────────
    results = {
        "users_included_in_compute": [
            {"user_id": s.user_id} for s in submissions
        ],
        "private_query_texts_by_user_id": {
            str(s.user_id): s.query_text for s in submissions
        },
        "private_liked_item_ids_by_user_id": {
            str(s.user_id): list(s.liked_catalog_item_ids or [])
            for s in submissions
        },
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "scoring": {
            "algorithm": "groupfit_pos_text",
            "alpha": GROUPFIT_ALPHA,
            "lambda": 0.0,
            "formula": "(1-alpha)*min_u max_j(e_i . liked_u_j) + alpha*mean_u(e_i . t_u)",
        },
        "applied_host_constraints": {
            "allowed_year_start": room.hard_constraint_year_start,
            "allowed_year_end": room.hard_constraint_year_end,
            "allowed_types": allowed_types,
        },
        "eligible_catalog_subset_size": int(eligible_count),
        "candidate_anime_ids_by_user_id": candidate_ids_by_user_id,
        "deduplicated_room_candidate_pool_ids": [item.id for item in candidate_items],
        "chosen_k": int(kmeans_result.k),
        "kmeans_objective": round(float(kmeans_result.objective), 4),
        "kmeans_silhouette": (
            round(float(kmeans_result.silhouette), 4)
            if kmeans_result.silhouette is not None
            else None
        ),
        "cluster_assignments": {
            str(item.id): int(assignment)
            for item, assignment in zip(candidate_items, kmeans_result.assignments, strict=True)
        },
        "group_match_scores_by_item_id": {
            str(item_id): round(score, 4)
            for item_id, score in scores_by_item_id.items()
        },
        "clusters": clusters_raw,
        "final_recommendations": final_recommendations,
        "vote_result_summary": [],
    }

    return results


def _safe_score(value: object) -> float:
    """Coerce a possibly-missing/None group_match_score to a sortable float."""
    if value is None:
        return 0.0
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(as_float):
        return 0.0
    return as_float


def compute_vote_summary(results_json: dict, votes_by_item_id: Counter[int]) -> list[dict]:
    """
    Build the final vote-summary list. Sort: votes desc, score desc, title asc.

    Defensive against malformed final_recommendations entries (missing id,
    None group_match_score, missing/None title) so a single corrupted item
    cannot break the post-vote summary for the whole room.
    """
    final_items = results_json.get("final_recommendations") or []
    summary: list[dict] = []

    for item in final_items:
        if not isinstance(item, dict):
            logger.warning("compute_vote_summary: skipping non-dict item: %r", item)
            continue

        raw_id = item.get("catalog_item_id")
        try:
            item_id = int(raw_id)
        except (TypeError, ValueError):
            logger.warning("compute_vote_summary: skipping item with bad id: %r", raw_id)
            continue

        vote_count = int(votes_by_item_id.get(item_id, 0))
        summary.append({**item, "vote_count": vote_count})

    summary.sort(
        key=lambda item: (
            -int(item.get("vote_count", 0)),
            -_safe_score(item.get("group_match_score")),
            str(item.get("title") or "").lower(),
        )
    )

    max_votes = summary[0]["vote_count"] if summary else 0
    for item in summary:
        item["is_winner"] = item["vote_count"] == max_votes and max_votes > 0

    return summary
