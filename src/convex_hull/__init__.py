"""Convex hull approximation solvers."""

from convex_hull.approximation import (
    ApproxConvexHull,
    approx_convex_hull,
    mvee_vmf_hull,
    uniform_sample_hull,
)
from convex_hull.geometry import Ellipsoid
from convex_hull.solvers.extents import ConvexHullviaExtents
from convex_hull.solvers.mvee import ConvexHullviaMVEE

__all__ = [
    "ApproxConvexHull",
    "ConvexHullviaDeepHull",
    "ConvexHullviaExtents",
    "ConvexHullviaMVEE",
    "Ellipsoid",
    "approx_convex_hull",
    "mvee_vmf_hull",
    "uniform_sample_hull",
]


def __getattr__(name):
    if name == "ConvexHullviaDeepHull":
        from convex_hull.solvers.deephull import ConvexHullviaDeepHull

        return ConvexHullviaDeepHull
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
