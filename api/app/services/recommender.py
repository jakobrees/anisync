from collections import Counter, defaultdict

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.embeddings import embed_texts
from app.ml.kmeans import choose_k_and_cluster
from app.models import CatalogItem, Room, RoomPreferenceSubmission


def item_tags(item: CatalogItem) -> list[str]:
    """
    Return anime tags safely.

    New rows use item.tags_json.
    Older rows can fall back to metadata_json["tags"].
    """
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

    This does not expose user-specific similarity scores or private retrieval data.
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
    """
    Derive a simple tag-based cluster label.

    Weighted tag frequency:
    tag score += max(group match score, 0)
    """
    weights: defaultdict[str, float] = defaultdict(float)

    for item in items:
        tags = item_tags(item)
        for tag in tags:
            weights[str(tag).lower()] += max(scores_by_id.get(item.id, 0.0), 0.0)

    if not weights:
        return ""

    top_tags = sorted(weights.items(), key=lambda pair: (-pair[1], pair[0]))[:2]
    return " / ".join(tag for tag, _ in top_tags)


def compute_recommendations(db: Session, room: Room) -> dict:
    """
    Full AniSync recommendation pipeline.

    This function implements the required math:
    1. collect private submissions
    2. embed query texts
    3. apply host hard constraints before retrieval
    4. retrieve top 100 per user from eligible subset
    5. union/deduplicate candidates
    6. choose K with bounded silhouette logic
    7. run manual K-means
    8. rank by group match score
    9. build top-5 cluster lists and top-2-per-cluster final list
    """
    submissions = list(
        db.scalars(
            select(RoomPreferenceSubmission)
            .where(RoomPreferenceSubmission.room_id == room.id)
            .order_by(RoomPreferenceSubmission.user_id)
        )
    )

    submissions = [submission for submission in submissions if submission.query_text.strip()]

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

    query_texts = [submission.query_text.strip() for submission in submissions]
    query_embeddings = embed_texts(query_texts)

    candidate_ids_by_user_id: dict[str, list[int]] = {}
    candidate_items_by_id: dict[int, CatalogItem] = {}

    for submission, query_embedding in zip(submissions, query_embeddings, strict=True):
        # pgvector exact cosine distance search inside host constraints.
        distance_expr = CatalogItem.embedding.cosine_distance(query_embedding.astype(float).tolist())

        rows = list(
            db.execute(
                select(CatalogItem, distance_expr.label("distance"))
                .where(
                    CatalogItem.year >= room.hard_constraint_year_start,
                    CatalogItem.year <= room.hard_constraint_year_end,
                    CatalogItem.media_type.in_(allowed_types),
                )
                .order_by(distance_expr.asc())
                .limit(100)
            )
        )

        item_ids: list[int] = []
        for item, _distance in rows:
            candidate_items_by_id[item.id] = item
            item_ids.append(item.id)

        candidate_ids_by_user_id[str(submission.user_id)] = item_ids

    candidate_items = list(candidate_items_by_id.values())

    if len(candidate_items) < 2:
        raise ValueError("The combined candidate pool is too small to produce recommendations.")

    candidate_embeddings = np.array(
        [np.array(item.embedding, dtype=np.float32) for item in candidate_items],
        dtype=np.float32,
    )

    kmeans_result = choose_k_and_cluster(candidate_embeddings)

    # GroupMatch(i) = average_u q_u^T e_i
    group_match_scores = query_embeddings @ candidate_embeddings.T
    average_scores = group_match_scores.mean(axis=0)
    scores_by_item_id = {
        item.id: float(score)
        for item, score in zip(candidate_items, average_scores, strict=True)
    }

    clusters_raw: list[dict] = []
    for cluster_index in range(kmeans_result.k):
        member_indexes = np.where(kmeans_result.assignments == cluster_index)[0].tolist()
        cluster_items = [candidate_items[index] for index in member_indexes]

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

    # Private data is stored in results_json for reproducibility,
    # but main.py sanitizes it before sending room data to clients.
    results = {
        "users_included_in_compute": [
            {
                "user_id": submission.user_id,
            }
            for submission in submissions
        ],
        "private_query_texts_by_user_id": {
            str(submission.user_id): submission.query_text
            for submission in submissions
        },
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
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


def compute_vote_summary(results_json: dict, votes_by_item_id: Counter[int]) -> list[dict]:
    """
    Build final result summary after every room member has voted.

    Sort order:
    1. vote count descending
    2. group match score descending
    3. title ascending
    """
    final_items = results_json.get("final_recommendations", [])
    summary: list[dict] = []

    for item in final_items:
        item_id = int(item["catalog_item_id"])
        vote_count = int(votes_by_item_id.get(item_id, 0))
        summary.append(
            {
                **item,
                "vote_count": vote_count,
            }
        )

    summary.sort(
        key=lambda item: (
            -item["vote_count"],
            -float(item.get("group_match_score", 0.0)),
            item["title"].lower(),
        )
    )

    max_votes = summary[0]["vote_count"] if summary else 0
    for item in summary:
        item["is_winner"] = item["vote_count"] == max_votes and max_votes > 0

    return summary
