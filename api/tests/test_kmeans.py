import numpy as np

from app.ml.kmeans import choose_k_and_cluster, manual_kmeans, silhouette_score


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
