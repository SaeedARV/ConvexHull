import numpy as np

from convex_hull.approximation._mvee import khachiyan_mvee
from convex_hull.approximation._sampling import sample_uniform_sphere, sample_vmf_base


def test_mvee_contains_points():
    rng = np.random.default_rng(0)
    points = rng.standard_normal(size=(50, 3))
    ellipsoid = khachiyan_mvee(points, tol=1e-6, max_iter=500)

    assert all(ellipsoid.contains(point, tol=1e-6) for point in points)


def test_project_and_normal():
    rng = np.random.default_rng(1)
    points = rng.standard_normal(size=(30, 3))
    ellipsoid = khachiyan_mvee(points, tol=1e-6, max_iter=500)
    boundary = ellipsoid.project_to_boundary(points[0])
    value = (boundary - ellipsoid.center) @ ellipsoid.A @ (boundary - ellipsoid.center)
    normal = ellipsoid.normal_vector(boundary)

    assert np.isclose(value, 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6)


def test_vmf_base_sampling():
    rng = np.random.default_rng(0)
    base = sample_vmf_base(64, 4, 5.0, rng)
    norms = np.linalg.norm(base, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-6)


def test_uniform_sphere_norms():
    rng = np.random.default_rng(0)
    samples = sample_uniform_sphere(128, 5, rng)
    norms = np.linalg.norm(samples, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-6)
