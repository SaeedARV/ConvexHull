"""
Benchmark script comparing convex hull solvers: Extents, MVEE, and DeepHull.

This utility generates synthetic point clouds, evaluates each solver against
SciPy's exact convex hull, and reports accuracy/runtime metrics suitable for
academic reporting.
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ConvexHullviaExtents import ConvexHullviaExtents
from ConvexHullviaMVEE import ConvexHullviaMVEE

try:
    from ConvexHullviaDeepHull import ConvexHullviaDeepHull
except ModuleNotFoundError:
    ConvexHullviaDeepHull = None  # type: ignore[assignment]


ArrayLike = Sequence[Sequence[float]]


@dataclass
class Metrics:
    vertex_precision: float
    vertex_recall: float
    vertex_f1: float
    hull_area_ratio: float
    coverage_rate: float
    runtime_ms: float
    predicted_vertices: int


def _sanitize_indices(indices: Sequence[int], n_points: int) -> List[int]:
    seen = set()
    valid: List[int] = []
    for idx in indices:
        j = int(idx)
        if 0 <= j < n_points and j not in seen:
            seen.add(j)
            valid.append(j)
    return valid


def _order_indices(points: np.ndarray, indices: List[int]) -> List[int]:
    if len(indices) < 3:
        return indices
    coords = points[indices]
    try:
        hull = ConvexHull(coords)
    except QhullError:
        return indices
    ordered = [indices[i] for i in hull.vertices]
    return ordered


def _points_to_indices(points: np.ndarray, selected: Sequence[Sequence[float]]) -> List[int]:
    lookup: Dict[Tuple[float, float], List[int]] = {}
    for idx, pt in enumerate(points):
        lookup.setdefault(tuple(pt), []).append(idx)
    mapped: List[int] = []
    for coords in selected:
        key = tuple(coords)
        if key not in lookup or not lookup[key]:
            raise ValueError("Selected point not found in original set.")
        mapped.append(lookup[key].pop())
    return mapped


def _convex_hull(points: np.ndarray) -> Optional[ConvexHull]:
    if points.shape[0] < 3:
        return None
    try:
        return ConvexHull(points)
    except QhullError:
        return None


def _points_inside(hull: ConvexHull, pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    a = hull.equations[:, :-1]
    b = hull.equations[:, -1]
    values = a.dot(pts.T) + b[:, np.newaxis]
    return (values <= tol).all(axis=0)


def evaluate_prediction(points: np.ndarray, gt_hull: ConvexHull, candidate_indices: Sequence[int]) -> Metrics:
    n_points = points.shape[0]
    indices = _sanitize_indices(candidate_indices, n_points)
    indices = _order_indices(points, indices)
    gt_vertices = gt_hull.vertices.tolist()
    gt_set = set(gt_vertices)
    pred_set = set(indices)

    if pred_set:
        true_positive = len(gt_set & pred_set)
        precision = true_positive / len(pred_set)
    else:
        precision = 0.0
    recall = len(gt_set & pred_set) / len(gt_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    predicted_hull = _convex_hull(points[indices]) if indices else None
    if predicted_hull is None:
        area_ratio = math.nan
        coverage = 0.0
    else:
        pred_area = predicted_hull.volume
        area_ratio = pred_area / gt_hull.volume if gt_hull.volume > 0 else math.nan
        inside_mask = _points_inside(predicted_hull, points)
        coverage = float(np.mean(inside_mask))

    return Metrics(
        vertex_precision=precision,
        vertex_recall=recall,
        vertex_f1=f1,
        hull_area_ratio=area_ratio,
        coverage_rate=coverage,
        runtime_ms=0.0,  # filled later
        predicted_vertices=len(indices),
    )


def run_extents(points: np.ndarray, random_samples: int = 2048) -> List[int]:
    solver = ConvexHullviaExtents(points)
    _, approx_set = solver.get_random_extents(random_samples, return_approx_hull=True)
    if not approx_set:
        return []
    approx_points = np.array(list(approx_set))
    approx_indices = _points_to_indices(points, approx_points)
    ordered = _order_indices(points, approx_indices)
    return ordered


def run_mvee(points: np.ndarray, **kwargs) -> List[int]:
    solver = ConvexHullviaMVEE(points)
    result = solver.compute(return_extents=True, **kwargs)
    approx = result[0] if isinstance(result, tuple) else result
    if approx.shape[0] < 3:
        return []
    approx_indices = _points_to_indices(points, approx)
    ordered = _order_indices(points, approx_indices)
    return ordered


def run_deephull(points: np.ndarray, model: Optional[ConvexHullviaDeepHull]) -> List[int]:
    if model is None:
        raise RuntimeError("ConvexHullviaDeepHull is unavailable (PyTorch not installed).")
    return model.predict_indices(points)


def generate_uniform(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    return rng.random((n, dim))


def generate_gaussian(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    pts = rng.normal(loc=0.0, scale=0.6, size=(n, dim))
    scales = np.linspace(0.5, 1.8, dim)
    cov = np.diag(scales)
    return pts @ cov


def generate_annulus(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    raw = rng.normal(size=(n, dim))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
    radii = rng.uniform(0.4, 1.0, size=(n, 1))
    shell = raw * radii
    noise = rng.normal(scale=0.05, size=(n, dim))
    return shell + noise


DATASET_GENERATORS: Dict[str, Callable[[np.random.Generator, int, int], np.ndarray]] = {
    "uniform": generate_uniform,
    "gaussian": generate_gaussian,
    "annulus": generate_annulus,
}


def iterate_point_sets(
    rng: np.random.Generator,
    dataset: str,
    n_samples: int,
    min_points: int,
    max_points: int,
    dimension: int,
) -> Iterable[np.ndarray]:
    generator = DATASET_GENERATORS[dataset]
    for _ in range(n_samples):
        n = rng.integers(min_points, max_points + 1)
        yield generator(rng, int(n), dimension)


def summarize(values: List[float]) -> str:
    filtered = [v for v in values if not math.isnan(v)]
    if not filtered:
        return "n/a"
    mean = statistics.mean(filtered)
    std = statistics.pstdev(filtered) if len(filtered) > 1 else 0.0
    return f"{mean:.3f} ± {std:.3f}"


def format_table(rows: List[List[str]], headers: List[str]) -> str:
    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    def fmt(row: List[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
    sep = "-+-".join("-" * w for w in col_widths)
    lines = [fmt(headers), sep]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def benchmark(
    dataset_names: Sequence[str],
    n_samples: int,
    min_points: int,
    max_points: int,
    seed: int,
    random_extents: int,
    mvee_kwargs: Dict[str, int],
    deephull_kwargs: Optional[Dict[str, float]] = None,
    dimension: int = 2,
) -> None:
    rng = np.random.default_rng(seed)
    if ConvexHullviaDeepHull is not None:
        deephull_model = ConvexHullviaDeepHull(**(deephull_kwargs or {}))
    else:
        deephull_model = None

    method_fns = {}
    
    method_fns["Extents"] = lambda pts: run_extents(pts, random_samples=random_extents)
    method_fns["MVEE"] = lambda pts: run_mvee(pts, **mvee_kwargs)
    if deephull_model is not None:
        method_fns["DeepHull"] = lambda pts: run_deephull(pts, deephull_model)

    for dataset in dataset_names:
        per_method: Dict[str, List[Metrics]] = {name: [] for name in method_fns}
        for points in iterate_point_sets(rng, dataset, n_samples, min_points, max_points, dimension):
            gt_hull = ConvexHull(points)
            for method_name, predict_fn in method_fns.items():
                start = time.perf_counter()
                try:
                    candidate = predict_fn(points)
                except Exception as exc:
                    print(f"[WARN] {method_name} failed on dataset '{dataset}': {exc}")
                    continue
                runtime_ms = (time.perf_counter() - start) * 1000.0
                metrics = evaluate_prediction(points, gt_hull, candidate)
                metrics.runtime_ms = runtime_ms
                per_method[method_name].append(metrics)

        headers = ["Method", "Vertex F1", "Precision", "Recall", "Area ratio", "Coverage", "Vertices", "Runtime (ms)"]
        rows: List[List[str]] = []
        for method_name, records in per_method.items():
            if not records:
                continue
            rows.append([
                method_name,
                summarize([m.vertex_f1 for m in records]),
                summarize([m.vertex_precision for m in records]),
                summarize([m.vertex_recall for m in records]),
                summarize([m.hull_area_ratio for m in records]),
                summarize([m.coverage_rate for m in records]),
                summarize([float(m.predicted_vertices) for m in records]),
                summarize([m.runtime_ms for m in records]),
            ])
        print(f"\n=== Dataset: {dataset} ===")
        if rows:
            print(format_table(rows, headers))
        else:
            print("No successful runs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare convex hull solvers on synthetic datasets.")
    parser.add_argument("--datasets", type=str, default="uniform,gaussian,annulus",
                        help="Comma-separated list of dataset types to evaluate.")
    parser.add_argument("--samples", type=int, default=200, help="Number of point clouds per dataset.")
    parser.add_argument("--min-points", type=int, default=20, help="Minimum number of points per instance.")
    parser.add_argument("--max-points", type=int, default=60, help="Maximum number of points per instance.")
    parser.add_argument("--seed", type=int, default=41, help="Random seed.")
    parser.add_argument("--random-extents", type=int, default=2048,
                        help="Number of random directions used by the Extents method.")
    parser.add_argument("--mvee-m", type=int, default=8, help="Number of directions sampled for MVEE.")
    parser.add_argument("--mvee-kappa", type=int, default=45, help="Concentration parameter for MVEE sampling.")
    parser.add_argument("--dimension", type=int, default=2,
                        help="Ambient dimensionality of the generated point clouds.")
    parser.add_argument("--deephull-epochs", type=int, default=200, help="Training epochs for DeepHull.")
    parser.add_argument("--deephull-lambda", type=float, default=2.0, help="Negative sample weight for DeepHull.")
    parser.add_argument("--deephull-epsilon", type=float, default=0.05,
                        help="Tolerance used to collect points near the learned decision boundary.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_list = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = [name for name in dataset_list if name not in DATASET_GENERATORS]
    if unknown:
        raise SystemExit(f"Unknown dataset types: {', '.join(unknown)}")
    benchmark(
        dataset_names=dataset_list,
        n_samples=args.samples,
        min_points=args.min_points,
        max_points=args.max_points,
        seed=args.seed,
        random_extents=args.random_extents,
        mvee_kwargs={"m": args.mvee_m, "kappa": args.mvee_kappa},
        deephull_kwargs={
            "max_epochs": args.deephull_epochs,
            "lambda_neg": args.deephull_lambda,
            "level_set_epsilon": args.deephull_epsilon,
        },
        dimension=args.dimension,
    )
