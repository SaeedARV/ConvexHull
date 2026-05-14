"""Convex hull approximation solvers."""

from convex_hull.geometry import Ellipsoid
from convex_hull.solvers.extents import ConvexHullviaExtents
from convex_hull.solvers.mvee import ConvexHullviaMVEE

__all__ = [
    "ConvexHullviaDeepHull",
    "ConvexHullviaExtents",
    "ConvexHullviaMVEE",
    "Ellipsoid",
]


def __getattr__(name):
    if name == "ConvexHullviaDeepHull":
        from convex_hull.solvers.deephull import ConvexHullviaDeepHull

        return ConvexHullviaDeepHull
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
