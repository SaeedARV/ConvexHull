import numpy as np

from convex_hull import ApproxConvexHull


def _try_backend(name, points, seed):
    try:
        return ApproxConvexHull(points, method="uniform", m=12, backend=name, random_state=seed)
    except Exception:
        return None


def test_backends_consistency_uniform():
    rng = np.random.default_rng(0)
    points = rng.standard_normal(size=(40, 3))

    hull_python = _try_backend("python", points, 0)
    assert hull_python is not None

    hull_numba = _try_backend("numba", points, 0)
    hull_cpp = _try_backend("cpp", points, 0)

    ref = set(hull_python.vertices.tolist())
    for hull in [hull_numba, hull_cpp]:
        if hull is None:
            continue
        assert set(hull.vertices.tolist()) == ref


def test_auto_backend_returns_vertices():
    points = np.random.default_rng(1).standard_normal(size=(30, 4))
    hull = ApproxConvexHull(points, method="uniform", m=16, backend="auto", random_state=1)

    assert hull.vertices.size > 0
