import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from kmeans_pca import (
    HERE,
    INITIAL_CENTROIDS,
    elbow_curve,
    kmeans,
    load_points,
    main,
    pca_first_component,
    squared_distances,
)

POINTS = HERE / "points.csv"

# Values printed by the ORIGINAL hw4_kmean.py (before the refactor) on points.csv.
ORIGINAL_SIZES = [636, 585, 1463, 1222, 1094]
ORIGINAL_CENTROIDS = np.array(
    [
        [27.266774030930158, -1.088484815183697],
        [-32.27032271684726, 2.6598414911912944],
        [-1.4540119217017722, -2.2393291816313843],
        [11.487377594398366, -1.7089834442426803],
        [-15.058957720669701, -0.4728141516667429],
    ]
)
ORIGINAL_GLOBAL_PC1 = np.array([0.9983831723, -0.0568422495])


@pytest.fixture(scope="module")
def points():
    return load_points(POINTS)


def test_load_points_shape(points):
    assert points.shape == (5000, 2)  # the trailing blank line of the file is ignored


def test_squared_distances_matches_definition():
    pts = np.array([[0.0, 0.0], [3.0, 4.0]])
    cen = np.array([[0.0, 0.0], [3.0, 0.0]])
    assert squared_distances(pts, cen).tolist() == [[0.0, 9.0], [25.0, 16.0]]


def test_kmeans_reproduces_original_script_output(points):
    result = kmeans(points)
    assert np.bincount(result.labels).tolist() == ORIGINAL_SIZES
    assert np.allclose(result.centroids, ORIGINAL_CENTROIDS, atol=1e-9)
    assert result.n_iter < 100


def test_kmeans_matches_scikit_learn_lloyd(points):
    ours = kmeans(points, tol=1e-12)
    ref = KMeans(
        n_clusters=5, init=INITIAL_CENTROIDS, n_init=1, algorithm="lloyd", tol=0, max_iter=300
    ).fit(points)
    assert np.array_equal(ours.labels, ref.labels_)
    assert np.allclose(ours.centroids, ref.cluster_centers_, atol=1e-6)
    assert ours.inertia == pytest.approx(ref.inertia_, rel=1e-8)


def test_kmeans_separates_two_blobs_and_inertia_is_consistent():
    rng = np.random.default_rng(0)
    a = rng.normal([-5, 0], 0.3, size=(50, 2))
    b = rng.normal([5, 0], 0.3, size=(50, 2))
    pts = np.vstack([a, b])
    res = kmeans(pts, np.array([[-1.0, 1.0], [1.0, -1.0]]))
    assert sorted(np.bincount(res.labels)) == [50, 50]
    assert len(set(res.labels[:50])) == 1 and len(set(res.labels[50:])) == 1
    expected = ((pts - res.centroids[res.labels]) ** 2).sum()
    assert res.inertia == pytest.approx(expected)


def test_empty_cluster_keeps_its_centroid():
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    init = np.array([[0.0, 0.0], [100.0, 100.0]])  # second centroid never wins a point
    res = kmeans(pts, init)
    assert np.allclose(res.centroids[1], [100.0, 100.0])
    assert set(res.labels) == {0}


def test_max_iter_is_respected():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(300, 2))
    res = kmeans(pts, np.array([[5.0, 5.0], [-5.0, -5.0], [5.0, -5.0]]), max_iter=2, tol=0)
    assert res.n_iter == 2


def test_pca_matches_scikit_learn_and_original(points):
    direction, mean, ratio = pca_first_component(points)
    ref = PCA(n_components=2).fit(points)
    assert abs(direction @ ref.components_[0]) == pytest.approx(1.0, abs=1e-9)
    assert ratio == pytest.approx(ref.explained_variance_ratio_[0])
    assert np.allclose(mean, points.mean(axis=0))
    assert np.allclose(direction, ORIGINAL_GLOBAL_PC1, atol=1e-9)
    assert np.linalg.norm(direction) == pytest.approx(1.0)


def test_pca_axis_of_an_elongated_cloud():
    rng = np.random.default_rng(2)
    t = rng.normal(0, 5, 500)
    pts = np.column_stack([t, 0.5 * t]) + rng.normal(0, 0.05, (500, 2))
    direction, _, ratio = pca_first_component(pts)
    assert np.allclose(direction, [2 / np.sqrt(5), 1 / np.sqrt(5)], atol=1e-2)
    assert ratio > 0.999


def test_elbow_curve_finds_the_true_k_on_separated_groups():
    rng = np.random.default_rng(3)
    centers = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    pts = np.vstack([rng.normal(c, 0.4, size=(60, 2)) for c in centers])
    rows = elbow_curve(pts, ks=range(1, 8))
    best_k = max((r for r in rows if r["silhouette"] is not None), key=lambda r: r["silhouette"])[
        "k"
    ]
    assert best_k == 4
    inertias = [r["inertia"] for r in rows]
    assert all(a > b for a, b in zip(inertias, inertias[1:], strict=False))


def test_y_only_weights_recover_five_groups(points):
    weights = np.array([0.0, 1.0])
    res = kmeans(points * weights, INITIAL_CENTROIDS * weights)
    assert sorted(np.bincount(res.labels).tolist()) == [997, 1000, 1000, 1000, 1003]


def test_cli_writes_figures(tmp_path):
    assert main(["--out", str(tmp_path), "--no-elbow"]) == 0
    assert (tmp_path / "kmeans_pca_clusters.png").stat().st_size > 5000
