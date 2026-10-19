"""K-means and PCA implemented from scratch with NumPy (coursework HW4).

The assignment: cluster a 2-D point cloud with K-means (k = 5, fixed starting centroids,
Euclidean distance), then run PCA on the whole data set and on every cluster to get the
direction of largest variance.

This is the original ``hw4_kmean.py`` with the same algorithm and the same default start
centroids, but vectorised, free of the hard-coded ``data`` path, and with the figures
actually saved. The old script's commented-out distance experiments (Mahalanobis-style
kernel, max-norm, ...) were dropped; they are in the git history.

Run ``python kmeans_pca.py --help`` for options.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# The start centroids used in the original assignment.
INITIAL_CENTROIDS = np.array([[10, 10], [-10, -10], [2, 2], [3, 3], [-3, -3]], dtype=float)


@dataclass
class KMeansResult:
    labels: np.ndarray  # (n,) cluster index per point
    centroids: np.ndarray  # (k, d)
    n_iter: int
    inertia: float  # sum of squared distances to the assigned centroid


def load_points(path: str | Path) -> np.ndarray:
    """Read a headerless comma-separated file of 2-D points (blank lines are ignored)."""
    return np.loadtxt(path, delimiter=",", ndmin=2)


def squared_distances(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(n, k) matrix of squared Euclidean distances."""
    diff = points[:, None, :] - centroids[None, :, :]
    return np.einsum("nkd,nkd->nk", diff, diff)


def kmeans(
    points: np.ndarray,
    init_centroids: np.ndarray = INITIAL_CENTROIDS,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> KMeansResult:
    """Lloyd's algorithm from explicit starting centroids.

    Stops when no centroid moves by more than ``tol`` (squared distance) or after
    ``max_iter`` passes. A cluster that loses all its points keeps its previous centroid,
    exactly like the original script.
    """
    points = np.asarray(points, dtype=float)
    centroids = np.array(init_centroids, dtype=float, copy=True)
    k = len(centroids)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # noqa: B007
        labels = squared_distances(points, centroids).argmin(axis=1)  # ties -> lowest index
        shift = 0.0
        for j in range(k):
            members = points[labels == j]
            if len(members) == 0:
                continue
            new = members.mean(axis=0)
            shift = max(shift, float(((new - centroids[j]) ** 2).sum()))
            centroids[j] = new
        if shift <= tol:
            break
    d2 = squared_distances(points, centroids)
    labels = d2.argmin(axis=1)
    return KMeansResult(labels, centroids, n_iter, float(d2[np.arange(len(points)), labels].sum()))


def pca_first_component(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Principal axis of a point set via the covariance matrix.

    Returns ``(direction, mean, explained_variance_ratio)``; ``direction`` is a unit vector
    whose first non-zero coordinate is positive (the sign of an eigenvector is arbitrary).
    """
    points = np.asarray(points, dtype=float)
    mean = points.mean(axis=0)
    cov = np.cov(points - mean, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order
    direction = eigenvectors[:, -1]
    nonzero = direction[np.abs(direction) > 1e-12]
    if len(nonzero) and nonzero[0] < 0:
        direction = -direction
    return direction, mean, float(eigenvalues[-1] / eigenvalues.sum())


def elbow_curve(points: np.ndarray, ks=range(1, 11), seed: int = 0) -> list[dict]:
    """Inertia and silhouette score of scikit-learn K-means for each k (model selection aid)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(points)
        sil = float(silhouette_score(points, km.labels_)) if k > 1 else None
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": sil})
    return rows


def plot_clusters(
    points, result: KMeansResult, path: str | Path, title: str = "", axes_scale: float = 2.0
):
    """Scatter plot of the clusters (``result.centroids`` in data coordinates) with PC1 axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(10, 5))
    g_dir, g_mean, g_ratio = pca_first_component(points)
    for j in range(len(result.centroids)):
        members = points[result.labels == j]
        ax.scatter(
            members[:, 0],
            members[:, 1],
            s=6,
            color=colors[j],
            alpha=0.55,
            label=f"cluster {j} (n={len(members)})",
        )
        if len(members) > 2:
            direction, mean, _ = pca_first_component(members)
            span = axes_scale * np.sqrt(np.var(members @ direction, ddof=1))
            ends = np.vstack([mean - span * direction, mean + span * direction])
            ax.plot(ends[:, 0], ends[:, 1], color="k", lw=2)
    ax.scatter(
        *result.centroids.T,
        marker="X",
        s=120,
        color="red",
        edgecolor="k",
        zorder=5,
        label="centroid",
    )
    span = 3 * np.sqrt(np.var(points @ g_dir, ddof=1))
    ends = np.vstack([g_mean - span * g_dir, g_mean + span * g_dir])
    ax.plot(
        ends[:, 0],
        ends[:, 1],
        color="grey",
        lw=1.5,
        ls="--",
        label=f"global PC1 ({g_ratio:.0%} var.)",
    )
    ax.set_title(title or "K-means with per-cluster principal axis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_elbow(rows: list[dict], path: str | Path, title: str = "Choosing k: elbow and silhouette"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(ks, [r["inertia"] for r in rows], "o-", color="#264653")
    ax1.set_xlabel("number of clusters k")
    ax1.set_ylabel("inertia (within-cluster SSE)", color="#264653")
    ax2 = ax1.twinx()
    sil = [(r["k"], r["silhouette"]) for r in rows if r["silhouette"] is not None]
    ax2.plot(*zip(*sil, strict=True), "s--", color="#e76f51")
    ax2.set_ylabel("silhouette score", color="#e76f51")
    ax2.grid(False)
    ax1.set_xticks(ks)
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=str(HERE / "points.csv"), help="CSV of 2-D points")
    parser.add_argument("--out", default=str(HERE / "results"), help="directory for figures")
    parser.add_argument(
        "--weights",
        nargs=2,
        type=float,
        default=(1.0, 1.0),
        metavar=("WX", "WY"),
        help="scale the x / y coordinates before clustering (default 1 1 = the assignment; "
        "0 1 clusters on y only)",
    )
    parser.add_argument("--tag", default="", help="suffix for the output file names")
    parser.add_argument("--no-elbow", action="store_true", help="skip the elbow/silhouette sweep")
    args = parser.parse_args(argv)

    points = load_points(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    weights = np.asarray(args.weights, dtype=float)
    weighted = points * weights

    fit = kmeans(weighted, INITIAL_CENTROIDS * weights)
    # report/plot centroids in the original coordinates (identical when weights are 1, 1)
    centroids = np.array(
        [points[fit.labels == j].mean(axis=0) for j in range(len(INITIAL_CENTROIDS))]
    )
    result = KMeansResult(fit.labels, centroids, fit.n_iter, fit.inertia)

    print(
        f"{len(points)} points, coordinate weights {args.weights[0]:g} {args.weights[1]:g}; K-means converged in "
        f"{result.n_iter} iterations, inertia {result.inertia:.1f}"
    )
    direction, _, ratio = pca_first_component(points)
    print(f"global PC1 direction {np.round(direction, 4)} ({ratio:.1%} of variance)")
    for j, centroid in enumerate(result.centroids):
        members = points[result.labels == j]
        d, _, r = pca_first_component(members)
        print(
            f"cluster {j}: n={len(members):4d} centroid=({centroid[0]:7.3f}, {centroid[1]:7.3f}) "
            f"PC1={np.round(d, 4)} ({r:.1%})"
        )
    title = "K-means (k=5) + per-cluster PC1"
    if tuple(args.weights) != (1.0, 1.0):
        title += f", coordinate weights (x, y) = ({args.weights[0]:g}, {args.weights[1]:g})"
    plot_clusters(points, result, out / f"kmeans_pca_clusters{tag}.png", title=title)
    if not args.no_elbow:
        rows = elbow_curve(weighted)
        plot_elbow(rows, out / f"elbow_silhouette{tag}.png")
        for r in rows:
            sil = "  -  " if r["silhouette"] is None else f"{r['silhouette']:.3f}"
            print(f"k={r['k']:2d} inertia={r['inertia']:12.1f} silhouette={sil}")
    print(f"figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
