import numpy as np
import pytest

from convex_hull import ApproxConvexHull, approx_convex_hull


def test_invalid_points():
    with pytest.raises(ValueError):
        ApproxConvexHull(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        ApproxConvexHull(np.array([[np.nan, 0.0]]))


def test_uniform_api_vertices():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    hull = ApproxConvexHull(points, method="uniform", m=4, random_state=0)

    assert hull.vertices.ndim == 1
    assert np.all(hull.vertices < points.shape[0])
    assert hull.get_vertices_points().shape[1] == 2


def test_functional_api():
    points = np.random.default_rng(0).standard_normal(size=(30, 3))
    hull = approx_convex_hull(points, method="uniform", m=10)

    assert hull.vertices.size > 0


def test_duplicate_points_mapping():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    hull = ApproxConvexHull(points, method="uniform", m=6, random_state=0)

    assert np.all(hull.vertices < points.shape[0])


def test_single_point():
    points = np.array([[1.0, 2.0]])
    hull = ApproxConvexHull(points, method="uniform", m=3, random_state=0)

    assert hull.vertices.size == 1
