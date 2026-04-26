from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KMeansResult:
    """Result object returned by manual K-means."""

    k: int
    assignments: np.ndarray
    centroids: np.ndarray
    objective: float
    iterations: int
    silhouette: float | None = None


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def squared_distances(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distance from each point to each centroid.

    Shape:
    - x: (N, D)
    - centroids: (K, D)
    - output: (N, K)
    """
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    c_sq = np.sum(centroids * centroids, axis=1, keepdims=True).T
    distances = x_sq + c_sq - 2.0 * (x @ centroids.T)
    return np.maximum(distances, 0.0)


def farthest_point_initialization(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Spread-aware initialization.

    1. Pick first point randomly.
    2. Pick each next centroid as the point farthest from its nearest existing centroid.
    """
    n = x.shape[0]
    if k <= 0:
        raise ValueError("k must be at least 1.")
    if n == 0:
        raise ValueError("Cannot initialize centroids on an empty dataset.")
    if k > n:
        raise ValueError("k cannot be larger than the number of points.")

    first_index = int(rng.integers(0, n))
    chosen = [first_index]

    while len(chosen) < k:
        current_centroids = x[np.array(chosen)]
        nearest_distance = squared_distances(x, current_centroids).min(axis=1)

        # Avoid choosing the same point twice.
        nearest_distance[np.array(chosen)] = -1.0
        next_index = int(np.argmax(nearest_distance))
        chosen.append(next_index)

    return x[np.array(chosen)].copy()


def run_kmeans_once(
    x: np.ndarray,
    k: int,
    *,
    seed: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    renormalize_centroids: bool = True,
) -> KMeansResult:
    """
    One K-means restart.

    This is a manual implementation:
    - no scikit-learn
    - squared Euclidean distance
    - farthest-point initialization
    - empty-cluster repair
    - convergence by centroid movement
    """
    if x.ndim != 2:
        raise ValueError("x must be a 2D matrix.")
    if x.shape[0] < k:
        raise ValueError("k cannot be larger than the number of points.")
    if k < 2:
        raise ValueError("k must be at least 2.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    rng = np.random.default_rng(seed)
    x = x.astype(np.float32)
    centroids = farthest_point_initialization(x, k, rng)

    assignments = np.zeros(x.shape[0], dtype=np.int64)
    # Pre-init `iteration` in case `max_iter == 1` and the loop body
    # branches early; without this the post-loop reference could NameError
    # on weirder inputs.
    iteration = 0

    for iteration in range(1, max_iter + 1):
        previous_centroids = centroids.copy()

        distances = squared_distances(x, centroids)
        assignments = np.argmin(distances, axis=1)

        # Repair empty clusters by moving the empty centroid to the point
        # with the largest current reconstruction error.
        counts = np.bincount(assignments, minlength=k)
        if np.any(counts == 0):
            reconstruction_errors = distances[np.arange(x.shape[0]), assignments]
            for empty_cluster in np.where(counts == 0)[0]:
                worst_point = int(np.argmax(reconstruction_errors))
                centroids[empty_cluster] = x[worst_point]
                assignments[worst_point] = empty_cluster
                reconstruction_errors[worst_point] = -1.0

        # Update centroids as cluster means.
        for cluster_index in range(k):
            members = x[assignments == cluster_index]
            if members.size > 0:
                centroids[cluster_index] = members.mean(axis=0)

        if renormalize_centroids:
            centroids = _normalize_rows(centroids)

        movement = np.linalg.norm(centroids - previous_centroids, axis=1).max()
        if movement < tol:
            break

    final_distances = squared_distances(x, centroids)
    final_objective = float(final_distances[np.arange(x.shape[0]), assignments].sum())

    return KMeansResult(
        k=k,
        assignments=assignments,
        centroids=centroids,
        objective=final_objective,
        iterations=iteration,
    )


def manual_kmeans(
    x: np.ndarray,
    k: int,
    *,
    n_init: int = 10,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_seed: int = 42,
) -> KMeansResult:
    """
    Run multiple restarts and return the lowest-objective K-means result.
    """
    best: KMeansResult | None = None

    for restart in range(n_init):
        result = run_kmeans_once(
            x,
            k,
            seed=random_seed + restart,
            max_iter=max_iter,
            tol=tol,
        )
        if best is None or result.objective < best.objective:
            best = result

    assert best is not None
    return best


def silhouette_score(x: np.ndarray, assignments: np.ndarray) -> float | None:
    """
    Compute average silhouette score manually.

    Returns None when the score is undefined (fewer than 2 clusters,
    or every point sits alone in its own cluster). The legal silhouette
    range is [-1.0, 1.0], so we use None as the "not computable" sentinel
    instead of a magic number that collides with a real worst-case score.

    This is O(N^2), which is fine because AniSync clusters only the
    room candidate pool, not the full anime catalog.
    """
    n = x.shape[0]
    labels = np.unique(assignments)

    if labels.size < 2 or labels.size >= n:
        return None

    pairwise = squared_distances(x, x)
    scores: list[float] = []

    for index in range(n):
        own_cluster = assignments[index]

        own_mask = assignments == own_cluster
        own_mask[index] = False

        if own_mask.sum() == 0:
            scores.append(0.0)
            continue

        a = float(pairwise[index, own_mask].mean())

        b_values: list[float] = []
        for other_cluster in labels:
            if other_cluster == own_cluster:
                continue
            other_mask = assignments == other_cluster
            if other_mask.sum() > 0:
                b_values.append(float(pairwise[index, other_mask].mean()))

        b = min(b_values) if b_values else 0.0
        denominator = max(a, b, 1e-12)
        scores.append((b - a) / denominator)

    return float(np.mean(scores))


def choose_k_and_cluster(
    x: np.ndarray,
    *,
    random_seed: int = 42,
    min_cluster_size: int = 5,
) -> KMeansResult:
    """
    Bounded silhouette-based K selection.

    Rules from the design document:
    - If N < 10, use K=2.
    - Else K candidates are 2..min(5, floor(N/8)).
    - Reject candidate K if any cluster has fewer than 5 items.
    - Choose the best silhouette score.
    - If scores are within 0.02, choose smaller K.
    - If no candidate passes, fallback to K=2.
    """
    n = x.shape[0]

    if n < 2:
        raise ValueError("Candidate pool must contain at least 2 items.")

    if n < 10:
        result = manual_kmeans(x, min(2, n), random_seed=random_seed)
        return KMeansResult(**{**result.__dict__, "silhouette": silhouette_score(x, result.assignments)})

    k_max = min(5, n // 8)
    k_values = list(range(2, max(2, k_max) + 1))

    valid_results: list[KMeansResult] = []

    for k in k_values:
        result = manual_kmeans(x, k, random_seed=random_seed + k)
        counts = np.bincount(result.assignments, minlength=k)

        if np.any(counts < min_cluster_size):
            continue

        sil = silhouette_score(x, result.assignments)
        valid_results.append(KMeansResult(**{**result.__dict__, "silhouette": sil}))

    if not valid_results:
        fallback = manual_kmeans(x, 2, random_seed=random_seed)
        return KMeansResult(**{**fallback.__dict__, "silhouette": silhouette_score(x, fallback.assignments)})

    # Use `is None` instead of truthy fallback so a legitimate silhouette of
    # 0.0 (boundary clustering) is not silently treated as the worst score.
    def score_or_worst(result: KMeansResult) -> float:
        return result.silhouette if result.silhouette is not None else -1.0

    best_score = max(score_or_worst(result) for result in valid_results)

    # Product-aware tie rule: if within 0.02, prefer smaller K.
    near_best = [
        result
        for result in valid_results
        if (best_score - score_or_worst(result)) <= 0.02
    ]

    near_best.sort(key=lambda result: (result.k, -score_or_worst(result)))
    return near_best[0]
