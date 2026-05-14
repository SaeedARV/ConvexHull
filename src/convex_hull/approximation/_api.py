from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ._mvee import Ellipsoid, khachiyan_mvee
from ._random import get_rng
from ._sampling import (
    apply_householder_to_rows,
    householder_vector,
    sample_uniform_sphere,
    sample_vmf_base,
)
from ._support import (
    SupportGapStats,
    argmax_dot_blocked,
    argmax_dot_with_ties,
    estimate_support_gap,
    gather_unique,
)
from ._validation import normalize_vertices, restore_original_indices, validate_points
from .backends import get_backend


@dataclass
class FacetInfo:
    hull: Any
    simplices: Optional[np.ndarray]
    neighbors: Optional[np.ndarray]
    equations: Optional[np.ndarray]
    coplanar: Optional[np.ndarray]
    good: Optional[np.ndarray]
    area: Optional[float]
    volume: Optional[float]


class ApproxConvexHull:
    """Approximate convex hull vertex set for a point cloud.

    Parameters mirror scipy.spatial.ConvexHull but return only an approximate
    vertex subset with probabilistic guarantees.
    """

    def __init__(
        self,
        points: np.ndarray,
        *,
        method: str = "uniform",
        backend: str = "auto",
        m: Optional[int] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        kappa: Optional[float] = None,
        random_state: Optional[np.random.Generator] = None,
        return_diagnostics: bool = False,
        compute_facets: bool = False,
        qhull_options: Optional[str] = None,
        check_input: bool = True,
        include_ties: bool = False,
        mvee_tol: float = 1e-7,
        mvee_max_iter: int = 1000,
        uniform_constant: float = 4.0,
    ) -> None:
        if check_input:
            validated = validate_points(points, deduplicate=True)
            pts = validated.points
        else:
            pts = np.ascontiguousarray(points, dtype=float)
            validated = None

        self.method = method
        self.backend = backend
        self.m = m
        self.epsilon = epsilon
        self.delta = delta
        self.kappa = kappa
        self.random_state = random_state
        self.return_diagnostics = return_diagnostics
        self.compute_facets = compute_facets
        self.qhull_options = qhull_options
        self.check_input = check_input
        self.include_ties = include_ties
        self.mvee_tol = mvee_tol
        self.mvee_max_iter = mvee_max_iter
        self.uniform_constant = uniform_constant

        rng = get_rng(random_state)
        timings: Dict[str, float] = {}

        if method == "uniform":
            vertices = _uniform_method(
                pts,
                rng=rng,
                m=m,
                epsilon=epsilon,
                delta=delta,
                backend=backend,
                include_ties=include_ties,
                constant=uniform_constant,
                timings=timings,
            )
        elif method == "mvee_vmf":
            if m is None or kappa is None:
                raise ValueError("m and kappa are required for method='mvee_vmf'")
            vertices = _mvee_vmf_method(
                pts,
                rng=rng,
                m=m,
                kappa=kappa,
                backend=backend,
                include_ties=include_ties,
                mvee_tol=mvee_tol,
                mvee_max_iter=mvee_max_iter,
                timings=timings,
            )
        else:
            raise ValueError("method must be 'uniform' or 'mvee_vmf'")

        if validated is not None:
            vertices = restore_original_indices(vertices, validated.unique_indices)
            original_points = validated.original_points
        else:
            original_points = pts

        self.points = original_points
        self.vertices = normalize_vertices(vertices)
        self._facet_info = None

        if compute_facets:
            self._facet_info = _compute_facets(self.points[self.vertices], qhull_options)

        self.diagnostics = timings if return_diagnostics else None

    def get_vertices_points(self) -> np.ndarray:
        return self.points[self.vertices]

    def support_error(self, num_probe: int = 512, *, random_state: Optional[np.random.Generator] = None) -> SupportGapStats:
        rng = get_rng(random_state)
        directions = sample_uniform_sphere(num_probe, self.points.shape[1], rng)
        subset = self.get_vertices_points()
        epsilon = self.epsilon if self.epsilon is not None else 0.0
        return estimate_support_gap(self.points, subset, directions, epsilon=epsilon)

    def to_scipy(self):
        if self._facet_info is None:
            raise ValueError("facets not computed; pass compute_facets=True")
        return self._facet_info.hull

    @property
    def simplices(self) -> Optional[np.ndarray]:
        return None if self._facet_info is None else self._facet_info.simplices

    @property
    def neighbors(self) -> Optional[np.ndarray]:
        return None if self._facet_info is None else self._facet_info.neighbors

    @property
    def equations(self) -> Optional[np.ndarray]:
        return None if self._facet_info is None else self._facet_info.equations

    @property
    def coplanar(self) -> Optional[np.ndarray]:
        return None if self._facet_info is None else self._facet_info.coplanar

    @property
    def good(self) -> Optional[np.ndarray]:
        return None if self._facet_info is None else self._facet_info.good

    @property
    def area(self) -> Optional[float]:
        return None if self._facet_info is None else self._facet_info.area

    @property
    def volume(self) -> Optional[float]:
        return None if self._facet_info is None else self._facet_info.volume

    def __repr__(self) -> str:
        return (
            f"ApproxConvexHull(n_points={self.points.shape[0]}, "
            f"n_vertices={self.vertices.shape[0]}, method='{self.method}', backend='{self.backend}')"
        )


def approx_convex_hull(points: np.ndarray, **kwargs: Any) -> ApproxConvexHull:
    return ApproxConvexHull(points, **kwargs)


def uniform_sample_hull(
    points: np.ndarray,
    *,
    m: Optional[int] = None,
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    backend: str = "auto",
    random_state: Optional[np.random.Generator] = None,
    **kwargs: Any,
) -> ApproxConvexHull:
    return ApproxConvexHull(
        points,
        method="uniform",
        backend=backend,
        m=m,
        epsilon=epsilon,
        delta=delta,
        random_state=random_state,
        **kwargs,
    )


def mvee_vmf_hull(
    points: np.ndarray,
    *,
    m: int,
    kappa: float,
    backend: str = "auto",
    random_state: Optional[np.random.Generator] = None,
    **kwargs: Any,
) -> ApproxConvexHull:
    return ApproxConvexHull(
        points,
        method="mvee_vmf",
        backend=backend,
        m=m,
        kappa=kappa,
        random_state=random_state,
        **kwargs,
    )


def _compute_uniform_m(epsilon: float, delta: float, d: int, constant: float) -> int:
    if epsilon <= 0.0 or delta <= 0.0 or delta >= 1.0:
        raise ValueError("epsilon must be >0 and delta must be in (0,1)")
    m = constant * (d / (epsilon**2)) * np.log(1.0 / delta)
    return int(np.ceil(m))


def _uniform_method(
    points: np.ndarray,
    *,
    rng: np.random.Generator,
    m: Optional[int],
    epsilon: Optional[float],
    delta: Optional[float],
    backend: str,
    include_ties: bool,
    constant: float,
    timings: Dict[str, float],
) -> np.ndarray:
    d = points.shape[1]
    if m is None:
        if epsilon is None or delta is None:
            raise ValueError("provide m or (epsilon, delta) for uniform method")
        m = _compute_uniform_m(epsilon, delta, d, constant)

    t0 = time.perf_counter()
    directions = sample_uniform_sphere(m, d, rng)
    timings["sample_directions"] = time.perf_counter() - t0

    if include_ties:
        max_vals, tie_indices = argmax_dot_with_ties(points, directions)
        winners = gather_unique(idx for group in tie_indices for idx in group)
        timings["argmax_dot"] = 0.0
        return winners

    t0 = time.perf_counter()
    backend_impl = get_backend(backend)
    winners = backend_impl.argmax_dot(points, directions)
    timings["argmax_dot"] = time.perf_counter() - t0

    return gather_unique(winners)


def _mvee_vmf_method(
    points: np.ndarray,
    *,
    rng: np.random.Generator,
    m: int,
    kappa: float,
    backend: str,
    include_ties: bool,
    mvee_tol: float,
    mvee_max_iter: int,
    timings: Dict[str, float],
) -> np.ndarray:
    d = points.shape[1]
    t0 = time.perf_counter()
    ellipsoid = khachiyan_mvee(points, tol=mvee_tol, max_iter=mvee_max_iter)
    timings["mvee"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    base_dirs = sample_vmf_base(m, d, kappa, rng)
    timings["sample_base_vmf"] = time.perf_counter() - t0

    backend_impl = get_backend(backend)
    all_winners = []
    e1 = np.zeros(d)
    e1[0] = 1.0

    for point in points:
        boundary = ellipsoid.project_to_boundary(point)
        normal = ellipsoid.normal_vector(boundary)
        if np.linalg.norm(normal) == 0.0:
            continue
        v_house = householder_vector(e1, normal)
        if hasattr(backend_impl, "apply_householder_to_rows"):
            dirs = backend_impl.apply_householder_to_rows(base_dirs, v_house)
        else:
            dirs = apply_householder_to_rows(base_dirs, v_house)
        if include_ties:
            _, tie_indices = argmax_dot_with_ties(points, dirs)
            for group in tie_indices:
                all_winners.extend(group.tolist())
        else:
            winners = backend_impl.argmax_dot(points, dirs)
            all_winners.extend(winners.tolist())

    return gather_unique(all_winners)


def _compute_facets(points: np.ndarray, qhull_options: Optional[str]) -> FacetInfo:
    try:
        from scipy.spatial import ConvexHull
    except ImportError as exc:
        raise ImportError("SciPy required for compute_facets") from exc

    hull = ConvexHull(points, qhull_options=qhull_options)
    return FacetInfo(
        hull=hull,
        simplices=hull.simplices,
        neighbors=hull.neighbors,
        equations=hull.equations,
        coplanar=hull.coplanar,
        good=hull.good,
        area=hull.area,
        volume=hull.volume,
    )
