import numpy as np
import pytest

from app.ml.kmeans import (
    choose_k_and_cluster,
    farthest_point_initialization,
    manual_kmeans,
    run_kmeans_once,
    silhouette_score,
)


def test_manual_kmeans_finds_two_simple_clusters():
    left = np.array([[0.0, 1.0], [0.1, 0.99], [-0.1, 0.98]], dtype=np.float32)
    right = np.array([[1.0, 0.0], [0.99, 0.1], [0.98, -0.1]], dtype=np.float32)
    x = np.vstack([left, right])

    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    result = manual_kmeans(x, 2, random_seed=1)

    assert result.k == 2
    assert set(result.assignments.tolist()) == {0, 1}
    assert result.objective >= 0


def test_silhouette_score_range():
    x = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ],
        dtype=np.float32,
    )
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    assignments = np.array([0, 0, 1, 1])

    score = silhouette_score(x, assignments)

    assert score is not None
    assert -1.0 <= score <= 1.0


def test_silhouette_score_returns_none_when_undefined():
    """
    Singleton clusters or a single global cluster cannot produce a
    meaningful silhouette. The function must return None so callers
    can distinguish "not computable" from a real score of -1.0.
    """
    x = np.eye(2, dtype=np.float32)

    assert silhouette_score(x, np.array([0, 1])) is None
    assert silhouette_score(x, np.array([0, 0])) is None


def test_choose_k_and_cluster_does_not_treat_zero_silhouette_as_missing():
    """
    Regression test for a bug where `silhouette or -1.0` collapsed a
    legitimate silhouette of 0.0 into the worst-possible score and
    therefore biased the K selection toward different K values.
    """
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=(+1.0, 0.0), scale=0.05, size=(20, 2))
    blob_b = rng.normal(loc=(-1.0, 0.0), scale=0.05, size=(20, 2))
    x = np.vstack([blob_a, blob_b]).astype(np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    result = choose_k_and_cluster(x)

    assert result.silhouette is not None
    assert result.silhouette > 0.5


def test_choose_k_and_cluster_small_pool_falls_back_to_two():
    x = np.eye(4, dtype=np.float32)
    result = choose_k_and_cluster(x)

    assert result.k == 2


def test_run_kmeans_once_rejects_invalid_max_iter():
    """
    Defensive guard: max_iter < 1 used to silently skip the loop and then
    NameError on `iteration` outside the loop. Now it must raise loudly.
    """
    x = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError):
        run_kmeans_once(x, k=2, seed=1, max_iter=0)


def test_run_kmeans_once_max_iter_one_does_not_crash():
    """Regression: max_iter=1 must produce a usable result, not NameError."""
    x = np.eye(4, dtype=np.float32)
    result = run_kmeans_once(x, k=2, seed=1, max_iter=1)
    assert result.k == 2
    assert result.assignments.shape == (4,)
    assert result.iterations >= 1


def test_run_kmeans_once_rejects_k_larger_than_n():
    x = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError):
        run_kmeans_once(x, k=3, seed=1)


def test_farthest_point_initialization_rejects_empty_dataset():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        farthest_point_initialization(np.zeros((0, 4), dtype=np.float32), k=1, rng=rng)


def test_farthest_point_initialization_rejects_zero_k():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        farthest_point_initialization(np.eye(4, dtype=np.float32), k=0, rng=rng)


def test_choose_k_and_cluster_rejects_too_few_points():
    """The recommender refuses to compute on a candidate pool of size < 2,
    but choose_k_and_cluster itself must also surface this clearly."""
    with pytest.raises(ValueError):
        choose_k_and_cluster(np.zeros((1, 4), dtype=np.float32))
