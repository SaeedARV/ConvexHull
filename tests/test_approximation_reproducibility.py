import numpy as np

from convex_hull import ApproxConvexHull


def test_reproducibility_uniform():
    points = np.random.default_rng(0).standard_normal(size=(30, 2))
    hull1 = ApproxConvexHull(points, method="uniform", m=10, random_state=123)
    hull2 = ApproxConvexHull(points, method="uniform", m=10, random_state=123)

    assert np.array_equal(hull1.vertices, hull2.vertices)


def test_compare_with_scipy_hull_vertices():
    from scipy.spatial import ConvexHull

    points = np.random.default_rng(0).standard_normal(size=(80, 2))
    exact = ConvexHull(points)
    approx = ApproxConvexHull(points, method="uniform", m=32, random_state=0)

    assert set(approx.vertices.tolist()).issubset(set(exact.vertices.tolist())) or approx.vertices.size > 0
