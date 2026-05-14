import numpy as np

from convex_hull import ConvexHullviaExtents


def test_extents_smoke():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )

    _, approx = ConvexHullviaExtents(points).get_random_extents(
        32,
        return_approx_hull=True,
        method="numpy",
    )

    assert approx
    assert all(len(point) == 2 for point in approx)
