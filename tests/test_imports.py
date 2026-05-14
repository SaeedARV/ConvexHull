def test_public_imports():
    from convex_hull import ConvexHullviaDeepHull, ConvexHullviaExtents, ConvexHullviaMVEE
    from convex_hull.geometry import Ellipsoid

    assert ConvexHullviaDeepHull is not None
    assert ConvexHullviaExtents is not None
    assert ConvexHullviaMVEE is not None
    assert Ellipsoid is not None
