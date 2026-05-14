import numpy as np

from convex_hull import ConvexHullviaMVEE
from convex_hull.geometry import Ellipsoid


def test_mvee_smoke():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )

    ellipsoid = Ellipsoid(points)
    approx = ConvexHullviaMVEE(points).compute(m=2, kappa=5)

    assert ellipsoid.center.shape == (2,)
    assert np.asarray(approx).size > 0
