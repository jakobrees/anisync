"""
Unit tests for app.services.recommender helpers.

These tests target the *defensive* layers added during the robustness audit:

- _safe_embedding: filters out None / wrong-shape / non-finite embeddings
  before they reach np.stack, where they would crash the entire compute.
- _safe_score: coerces a malformed group_match_score (None, "n/a", NaN)
  into a sortable float.
- compute_vote_summary: never raises on a malformed final_recommendations
  entry; instead skips it and still produces a valid summary for the rest.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.recommender import (
    _safe_embedding,
    _safe_score,
    compute_vote_summary,
)


EMBEDDING_DIM = 384


def _fake_item(item_id: int, embedding) -> SimpleNamespace:
    return SimpleNamespace(id=item_id, embedding=embedding)


# ─── _safe_embedding ─────────────────────────────────────────────────────


def test_safe_embedding_returns_array_for_valid_vector():
    raw = list(np.linspace(-1.0, 1.0, EMBEDDING_DIM))
    item = _fake_item(1, raw)
    result = _safe_embedding(item)
    assert result is not None
    assert result.shape == (EMBEDDING_DIM,)
    assert result.dtype == np.float32


def test_safe_embedding_rejects_none():
    assert _safe_embedding(_fake_item(1, None)) is None


def test_safe_embedding_rejects_wrong_dimension():
    """An item whose embedding has the wrong dim must be filtered, not crash np.stack."""
    too_short = [0.0] * 100
    too_long = [0.0] * (EMBEDDING_DIM + 1)
    assert _safe_embedding(_fake_item(1, too_short)) is None
    assert _safe_embedding(_fake_item(2, too_long)) is None


def test_safe_embedding_rejects_2d_blob():
    bad = np.zeros((2, EMBEDDING_DIM), dtype=np.float32)
    assert _safe_embedding(_fake_item(1, bad)) is None


def test_safe_embedding_rejects_nan_or_inf():
    """A vector with NaN/Inf would propagate through cosine sim and silently
    poison every score for the room. We drop it instead."""
    nan_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    nan_vec[0] = np.nan
    inf_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    inf_vec[0] = np.inf
    assert _safe_embedding(_fake_item(1, nan_vec)) is None
    assert _safe_embedding(_fake_item(2, inf_vec)) is None


def test_safe_embedding_rejects_non_numeric():
    assert _safe_embedding(_fake_item(1, "not a vector")) is None
    assert _safe_embedding(_fake_item(2, [object()])) is None


# ─── _safe_score ─────────────────────────────────────────────────────────


def test_safe_score_passes_through_floats():
    assert _safe_score(0.7) == pytest.approx(0.7)
    assert _safe_score(-0.3) == pytest.approx(-0.3)


def test_safe_score_handles_none_and_garbage():
    assert _safe_score(None) == 0.0
    assert _safe_score("n/a") == 0.0
    assert _safe_score(object()) == 0.0


def test_safe_score_replaces_non_finite_with_zero():
    assert _safe_score(float("nan")) == 0.0
    assert _safe_score(float("inf")) == 0.0
    assert _safe_score(float("-inf")) == 0.0


# ─── compute_vote_summary ────────────────────────────────────────────────


def test_compute_vote_summary_basic_winner():
    results = {
        "final_recommendations": [
            {"catalog_item_id": 1, "title": "Apple", "group_match_score": 0.5},
            {"catalog_item_id": 2, "title": "Banana", "group_match_score": 0.7},
            {"catalog_item_id": 3, "title": "Cherry", "group_match_score": 0.6},
        ]
    }
    votes = Counter({1: 3, 2: 1, 3: 0})

    summary = compute_vote_summary(results, votes)

    assert [item["catalog_item_id"] for item in summary] == [1, 2, 3]
    assert summary[0]["is_winner"] is True
    assert summary[1]["is_winner"] is False
    assert summary[2]["is_winner"] is False


def test_compute_vote_summary_no_votes_yields_no_winner():
    """If nobody has voted yet, no item should be flagged as winner."""
    results = {
        "final_recommendations": [
            {"catalog_item_id": 1, "title": "A", "group_match_score": 0.5},
            {"catalog_item_id": 2, "title": "B", "group_match_score": 0.6},
        ]
    }
    summary = compute_vote_summary(results, Counter())
    assert all(item["is_winner"] is False for item in summary)


def test_compute_vote_summary_ties_pick_winner_by_score_then_title():
    results = {
        "final_recommendations": [
            {"catalog_item_id": 1, "title": "Banana", "group_match_score": 0.5},
            {"catalog_item_id": 2, "title": "Apple", "group_match_score": 0.6},
        ]
    }
    votes = Counter({1: 2, 2: 2})
    summary = compute_vote_summary(results, votes)
    # Same vote count, score 0.6 wins → Apple first. Both flagged is_winner.
    assert summary[0]["catalog_item_id"] == 2
    assert all(item["is_winner"] for item in summary)


def test_compute_vote_summary_skips_malformed_entries_without_crashing():
    """
    Defensive: a single bad entry in final_recommendations (missing id, None
    score, non-dict) must not break the whole post-vote summary.
    """
    results = {
        "final_recommendations": [
            "a string snuck in",
            {"title": "missing id"},
            {"catalog_item_id": "not-an-int", "title": "bad id"},
            {"catalog_item_id": 7, "title": "OK", "group_match_score": None},
            {"catalog_item_id": 8, "title": "Also OK", "group_match_score": 0.4},
        ]
    }
    summary = compute_vote_summary(results, Counter({7: 2, 8: 1}))
    ids = [item["catalog_item_id"] for item in summary]
    assert ids == [7, 8]
    assert summary[0]["is_winner"] is True


def test_compute_vote_summary_handles_missing_key():
    """results_json without `final_recommendations` must not KeyError."""
    summary = compute_vote_summary({}, Counter())
    assert summary == []


def test_compute_vote_summary_handles_none_final_recommendations():
    summary = compute_vote_summary({"final_recommendations": None}, Counter())
    assert summary == []


def test_compute_vote_summary_handles_none_title_in_sort():
    """A row with title=None used to crash the str().lower() sort key."""
    results = {
        "final_recommendations": [
            {"catalog_item_id": 1, "title": None, "group_match_score": 0.5},
            {"catalog_item_id": 2, "title": "Z", "group_match_score": 0.5},
        ]
    }
    summary = compute_vote_summary(results, Counter({1: 1, 2: 1}))
    assert {item["catalog_item_id"] for item in summary} == {1, 2}
