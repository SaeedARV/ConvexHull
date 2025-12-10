#!/usr/bin/env python3
"""
End-to-end experimental harness for convex-hull solvers.

This script reproduces the synthetic, scalability, and downstream anomaly
experiments described in the research plan. It consumes the existing solver
implementations in this repository:
    - ConvexHullviaExtents
    - ConvexHullviaMVEE
    - ConvexHullviaDeepHull (optional; skipped gracefully if unavailable)

Key outputs
-----------
* CSV summaries of all numeric metrics.
* Publication-ready plots saved under the output directory.
* Lightweight visualisations for 2D/3D cases.

The script is modular: each experiment has a dedicated function and uses shared
helpers for dataset generation, hull execution, metric computation, and plotting.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

# Non-interactive backend so plots can be written without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import ConvexHull, QhullError  # noqa: E402

from ConvexHullviaExtents import ConvexHullviaExtents  # noqa: E402
from ConvexHullviaMVEE import ConvexHullviaMVEE  # noqa: E402

try:
    from ConvexHullviaDeepHull import ConvexHullviaDeepHull  # type: ignore
except ModuleNotFoundError:
    ConvexHullviaDeepHull = None  # type: ignore[assignment]
    warnings.warn("ConvexHullviaDeepHull is unavailable; DeepHull runs will be skipped.", RuntimeWarning)


# ---------------------------------------------------------------------------
# Dataset generators (reused from existing utilities)
# ---------------------------------------------------------------------------
def generate_uniform(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """Uniform samples on [0, 1]^d."""
    return rng.random((n, dim))


def generate_uniform_hypercube(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """Uniform samples on [-1, 1]^d (reuses the base uniform generator)."""
    base = generate_uniform(rng, n, dim)
    return 2.0 * base - 1.0


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


def generate_anisotropic_gaussian(rng: np.random.Generator, n: int, dim: int, condition: float) -> np.ndarray:
    """
    Gaussian stretched along each axis with a prescribed condition number.
    The diagonal entries span [1, condition] geometrically.
    """
    diag = np.geomspace(1.0, float(condition), num=dim)
    z = rng.normal(size=(n, dim))
    return z @ np.diag(diag)


DATASET_GENERATORS: Dict[str, Callable[[np.random.Generator, int, int], np.ndarray]] = {
    "uniform01": generate_uniform,
    "uniform_cube": generate_uniform_hypercube,
    "gaussian": generate_gaussian,
    "annulus": generate_annulus,
}


# ---------------------------------------------------------------------------
# Helper dataclasses and utility functions
# ---------------------------------------------------------------------------
ArrayLike = Sequence[Sequence[float]]


@dataclass
class MethodResult:
    method: str
    vertices: np.ndarray
    indices: List[int]
    runtime_ms: float
    status: str = "ok"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    import csv

    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    lookup: Dict[Tuple[float, ...], List[int]] = {}
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
    if points.shape[0] < max(3, points.shape[1] + 1):
        return None
    try:
        return ConvexHull(points)
    except QhullError:
        return None


def points_inside(hull: ConvexHull, pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    a = hull.equations[:, :-1]
    b = hull.equations[:, -1]
    values = a.dot(pts.T) + b[:, np.newaxis]
    return (values <= tol).all(axis=0)


def sample_directions(rng: np.random.Generator, num: int, dim: int) -> np.ndarray:
    vecs = rng.normal(size=(num, dim))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def support_function(points: np.ndarray, directions: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.full(directions.shape[0], -np.inf, dtype=np.float64)
    dots = points @ directions.T
    return np.max(dots, axis=0)


def estimate_volume_ratio(
    rng: np.random.Generator,
    true_hull: ConvexHull,
    approx_vertices: np.ndarray,
    n_samples: int = 5000,
) -> Tuple[float, float, float]:
    """
    Monte Carlo volume estimate for true and approximate hulls; returns
    (ratio, vol_true, vol_approx).
    """
    pts = true_hull.points
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = maxs - mins
    span[span == 0] = 1.0
    samples = rng.random((n_samples, pts.shape[1])) * span + mins
    inside_true = points_inside(true_hull, samples)

    approx_hull = _convex_hull(approx_vertices) if approx_vertices.size else None
    if approx_hull is None:
        inside_approx = np.zeros_like(inside_true, dtype=bool)
    else:
        inside_approx = points_inside(approx_hull, samples)

    box_volume = float(np.prod(span))
    vol_true = box_volume * float(np.mean(inside_true))
    vol_approx = box_volume * float(np.mean(inside_approx))
    ratio = vol_approx / vol_true if vol_true > 0 else math.nan
    return ratio, vol_true, vol_approx


def support_error(
    rng: np.random.Generator,
    true_hull: ConvexHull,
    approx_vertices: np.ndarray,
    directions: int = 2000,
) -> float:
    dirs = sample_directions(rng, directions, true_hull.points.shape[1])
    true_vertices = true_hull.points[true_hull.vertices]
    true_support = support_function(true_vertices, dirs)
    approx_support = support_function(approx_vertices, dirs)
    if np.all(~np.isfinite(approx_support)):
        return math.nan
    return float(np.max(np.abs(approx_support - true_support)))


def membership_accuracy(
    rng: np.random.Generator,
    true_hull: ConvexHull,
    approx_vertices: np.ndarray,
    n_samples: int = 5000,
) -> float:
    pts = true_hull.points
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = maxs - mins
    span[span == 0] = 1.0
    samples = rng.random((n_samples, pts.shape[1])) * span + mins
    true_labels = points_inside(true_hull, samples)
    approx_hull = _convex_hull(approx_vertices) if approx_vertices.size else None
    if approx_hull is None:
        pred_labels = np.zeros_like(true_labels, dtype=bool)
    else:
        pred_labels = points_inside(approx_hull, samples)
    return float(np.mean(true_labels == pred_labels))


def estimate_memory_bytes(vertices: np.ndarray) -> int:
    return int(vertices.nbytes)


def _convex_hull_worker(arr: np.ndarray) -> Optional[ConvexHull]:
    try:
        return ConvexHull(arr)
    except Exception:
        return None


def try_exact_hull_with_timeout(points: np.ndarray, timeout_s: float = 5.0) -> Tuple[Optional[ConvexHull], str]:
    """
    Attempt to compute an exact hull in a subprocess to avoid hard hangs.
    Returns (hull | None, status).
    """
    with ProcessPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_convex_hull_worker, points)
        try:
            hull = fut.result(timeout=timeout_s)
        except TimeoutError:
            return None, "timeout"
    return hull, "ok" if hull is not None else "failed"


# ---------------------------------------------------------------------------
# Solver wrappers
# ---------------------------------------------------------------------------
def run_extents(points: np.ndarray, random_samples: int = 2048) -> MethodResult:
    start = time.perf_counter()
    solver = ConvexHullviaExtents(points)
    _, approx_set = solver.get_random_extents(random_samples, return_approx_hull=True)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    if not approx_set:
        return MethodResult("Extents", np.empty((0, points.shape[1])), [], runtime_ms, status="empty")
    approx_vertices = np.array(list(approx_set))
    try:
        indices = _points_to_indices(points, approx_vertices)
        indices = _order_indices(points, indices)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"Extents index mapping failed: {exc}")
        indices = []
    return MethodResult("Extents", approx_vertices, indices, runtime_ms)


def run_mvee(points: np.ndarray, **kwargs: Any) -> MethodResult:
    start = time.perf_counter()
    solver = ConvexHullviaMVEE(points)
    result = solver.compute(return_extents=True, **kwargs)
    approx = result[0] if isinstance(result, tuple) else result
    runtime_ms = (time.perf_counter() - start) * 1000.0
    if approx.shape[0] < 3:
        return MethodResult("MVEE", np.empty((0, points.shape[1])), [], runtime_ms, status="empty")
    approx_vertices = np.asarray(approx)
    try:
        indices = _points_to_indices(points, approx_vertices)
        indices = _order_indices(points, indices)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"MVEE index mapping failed: {exc}")
        indices = []
    return MethodResult("MVEE", approx_vertices, indices, runtime_ms)


def run_deephull(points: np.ndarray, method: str, **kwargs: Any) -> MethodResult:
    start = time.perf_counter()
    if ConvexHullviaDeepHull is None:
        return MethodResult(f"DeepHull-{method}", np.empty((0, points.shape[1])), [], 0.0, status="missing")
    model = ConvexHullviaDeepHull(method=method, **kwargs)
    indices = model.predict_indices(points)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    approx_vertices = points[indices] if indices else np.empty((0, points.shape[1]))
    indices = _sanitize_indices(indices, points.shape[0])
    indices = _order_indices(points, indices)
    return MethodResult(f"DeepHull-{method}", approx_vertices, indices, runtime_ms)


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------
def compute_low_dim_metrics(
    rng: np.random.Generator,
    points: np.ndarray,
    true_hull: ConvexHull,
    method_result: MethodResult,
) -> Dict[str, Any]:
    support_err = support_error(rng, true_hull, method_result.vertices)
    vol_ratio, vol_true, vol_approx = estimate_volume_ratio(rng, true_hull, method_result.vertices)
    member_acc = membership_accuracy(rng, true_hull, method_result.vertices)
    return {
        "support_error": support_err,
        "volume_ratio": vol_ratio,
        "volume_true": vol_true,
        "volume_approx": vol_approx,
        "membership_acc": member_acc,
        "runtime_ms": method_result.runtime_ms,
        "n_vertices": int(method_result.vertices.shape[0]),
        "status": method_result.status,
    }


def pairwise_support_distribution(
    directions: np.ndarray,
    support_by_method: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (m1, vals1), (m2, vals2) in itertools.combinations(support_by_method.items(), 2):
        if vals1.shape != vals2.shape:
            continue
        diffs = np.abs(vals1 - vals2)
        for idx, diff in enumerate(diffs):
            if not np.isfinite(diff):
                continue
            rows.append({"pair": f"{m1} vs {m2}", "direction": int(idx), "support_diff": float(diff)})
    return rows


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_metric_vs_n(
    results: List[Dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    output_dir: str,
    logy: bool = False,
) -> None:
    ensure_dir(output_dir)
    by_dim: Dict[int, Dict[str, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in results:
        dim = int(row["dimension"])
        method = str(row["method"])
        n = int(row["n_points"])
        value = row.get(metric)
        if value is None or not np.isfinite(value):
            continue
        by_dim[dim][method].append((n, float(value)))

    for dim, method_map in by_dim.items():
        plt.figure(figsize=(6, 4))
        for method, pairs in method_map.items():
            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            xs, ys = zip(*pairs_sorted)
            plt.plot(xs, ys, marker="o", label=method)
        plt.xlabel("n")
        plt.ylabel(ylabel)
        plt.title(f"{title} (d={dim})")
        if logy:
            plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_d{dim}.png"), dpi=200)
        plt.close()


def plot_runtime_vs_dimension(
    results: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    ensure_dir(output_dir)
    by_n: Dict[int, Dict[str, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in results:
        n = int(row["n_points"])
        method = str(row["method"])
        dim = int(row["dimension"])
        runtime = row.get("runtime_ms")
        if runtime is None or not np.isfinite(runtime):
            continue
        by_n[n][method].append((dim, float(runtime)))

    for n_val, method_map in by_n.items():
        plt.figure(figsize=(6, 4))
        for method, pairs in method_map.items():
            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            xs, ys = zip(*pairs_sorted)
            plt.plot(xs, ys, marker="o", label=method)
        plt.xlabel("dimension")
        plt.ylabel("runtime (ms)")
        plt.yscale("log")
        plt.title(f"Runtime vs dimension (n={n_val})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"runtime_vs_dim_n{n_val}.png"), dpi=200)
        plt.close()


def plot_feasibility_heatmap(
    feasibility: Dict[Tuple[str, int, int], bool],
    output_path: str,
) -> None:
    methods = sorted({key[0] for key in feasibility})
    dims = sorted({key[1] for key in feasibility})
    ns = sorted({key[2] for key in feasibility})

    grid = np.zeros((len(methods), len(dims) * len(ns)))
    labels: List[str] = []
    for j, d in enumerate(dims):
        for k, n in enumerate(ns):
            labels.append(f"d={d}\nn={n}")
            col = j * len(ns) + k
            for i, m in enumerate(methods):
                grid[i, col] = 1.0 if feasibility.get((m, d, n), False) else 0.0

    plt.figure(figsize=(1.8 * len(ns), 0.6 * len(methods) + 2))
    im = plt.imshow(grid, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    plt.yticks(range(len(methods)), methods)
    plt.xticks(range(grid.shape[1]), labels)
    plt.colorbar(im, label="Completed within 60s")
    plt.title("Feasibility heatmap (high-dimensional)")
    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_violin_support(
    rows: List[Dict[str, Any]],
    output_path: str,
) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(output_path))
    pair_to_vals: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        pair_to_vals[row["pair"]].append(float(row["support_diff"]))
    labels = list(pair_to_vals.keys())
    data = [pair_to_vals[label] for label in labels]

    plt.figure(figsize=(max(6, 1.2 * len(labels)), 4))
    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
    plt.ylabel("|h_i - h_j|")
    plt.title("Pairwise support-function discrepancy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_anomaly_curves(
    roc_points: Dict[str, Tuple[np.ndarray, np.ndarray]],
    pr_points: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: str,
) -> None:
    ensure_dir(output_dir)
    plt.figure(figsize=(6, 4))
    for method, (fpr, tpr) in roc_points.items():
        plt.plot(fpr, tpr, label=method)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves (anomaly detection)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    for method, (recall, precision) in pr_points.items():
        plt.plot(recall, precision, label=method)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=200)
    plt.close()


def plot_bar_metrics(
    metrics: Dict[str, Tuple[float, float]],
    output_path: str,
) -> None:
    labels = list(metrics.keys())
    aurocs = [metrics[m][0] for m in labels]
    auprs = [metrics[m][1] for m in labels]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(max(6, 1.2 * len(labels)), 4))
    plt.bar(x - width / 2, aurocs, width, label="AUROC")
    plt.bar(x + width / 2, auprs, width, label="AUPR")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title("Anomaly detection summary")
    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_low_dim_visualisations(
    points: np.ndarray,
    true_hull: ConvexHull,
    approximations: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    dim = points.shape[1]
    ensure_dir(os.path.dirname(output_path))
    if dim == 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(points[:, 0], points[:, 1], s=5, alpha=0.4, label="points")
        for simplex in true_hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], "k-", alpha=0.4)
        for name, verts in approximations.items():
            if verts.size == 0:
                continue
            plt.plot(verts[:, 0], verts[:, 1], "o", label=f"{name} vertices")
        plt.legend()
        plt.title("2D hull comparison")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, alpha=0.25, label="points")
        for name, verts in approximations.items():
            if verts.size == 0:
                continue
            ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=25, label=f"{name} vertices")
        ax.set_title("3D hull comparison")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()


# ---------------------------------------------------------------------------
# Anomaly detection metrics (AUROC/AUPR without sklearn dependency)
# ---------------------------------------------------------------------------
def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Computes AUROC with the trapezoidal rule."""
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)
    tpr = tps / (tps[-1] if tps[-1] > 0 else 1)
    fpr = fps / (fps[-1] if fps[-1] > 0 else 1)
    return float(np.trapz(tpr, fpr))


def _precision_recall(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)
    denom = tps + fps
    precision = np.divide(tps, denom, out=np.ones_like(tps, dtype=float), where=denom > 0)
    recall = np.divide(tps, tps[-1] if tps[-1] > 0 else 1, out=np.zeros_like(tps, dtype=float), where=True)
    return precision, recall


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall = _precision_recall(labels, scores)
    # Ensure recall starts at 0 and ends at 1 for a proper integral
    recall_ext = np.concatenate([[0.0], recall, [1.0]])
    precision_ext = np.concatenate([[precision[0]], precision, [0.0]])
    return float(np.trapz(precision_ext, recall_ext))


def _roc_curve(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)
    tpr = tps / (tps[-1] if tps[-1] > 0 else 1)
    fpr = fps / (fps[-1] if fps[-1] > 0 else 1)
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return fpr, tpr


def _pr_curve(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    precision, recall = _precision_recall(labels, scores)
    precision = np.concatenate([[precision[0]], precision, [0.0]])
    recall = np.concatenate([[0.0], recall, [1.0]])
    return recall, precision


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------
def run_low_dim_experiments(
    rng: np.random.Generator,
    output_dir: str,
    deephull_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    dims = [2, 3, 5, 8]
    ns = [200, 500, 1000]
    results: List[Dict[str, Any]] = []

    plot_samples: Dict[int, Tuple[np.ndarray, ConvexHull, Dict[str, np.ndarray]]] = {}

    for dim in dims:
        for n in ns:
            for dataset_name in ["uniform_cube", "gaussian"]:
                points = DATASET_GENERATORS[dataset_name](rng, n, dim)
                true_hull = ConvexHull(points)
                method_outputs: List[MethodResult] = []
                method_outputs.append(run_extents(points, random_samples=2048))
                method_outputs.append(run_mvee(points, m=8, kappa=45))
                for method in ["original", "convex"]:
                    method_outputs.append(run_deephull(points, method=method, **deephull_kwargs))

                for cond in [1, 10, 100]:
                    ani_points = generate_anisotropic_gaussian(rng, n, dim, condition=cond)
                    ani_hull = ConvexHull(ani_points)
                    ani_outputs: List[MethodResult] = []
                    ani_outputs.append(run_extents(ani_points, random_samples=2048))
                    ani_outputs.append(run_mvee(ani_points, m=8, kappa=45))
                    for method in ["original", "convex"]:
                        ani_outputs.append(run_deephull(ani_points, method=method, **deephull_kwargs))
                    for mres in ani_outputs:
                        if mres.status == "missing":
                            continue
                        metrics = compute_low_dim_metrics(rng, ani_points, ani_hull, mres)
                        results.append(
                            {
                                "dimension": dim,
                                "n_points": n,
                                "dataset": f"anisotropic_cond{cond}",
                                "method": mres.method,
                                **metrics,
                            }
                        )

                for mres in method_outputs:
                    if mres.status == "missing":
                        continue
                    metrics = compute_low_dim_metrics(rng, points, true_hull, mres)
                    results.append(
                        {
                            "dimension": dim,
                            "n_points": n,
                            "dataset": dataset_name,
                            "method": mres.method,
                            **metrics,
                        }
                    )

                # Store samples for visualisation
                if dim in (2, 3) and (dim not in plot_samples):
                    approximations = {m.method: m.vertices for m in method_outputs if m.vertices.size}
                    plot_samples[dim] = (points, true_hull, approximations)

    ensure_dir(output_dir)
    save_csv(
        os.path.join(output_dir, "low_dim_results.csv"),
        results,
        fieldnames=[
            "dimension",
            "n_points",
            "dataset",
            "method",
            "support_error",
            "volume_ratio",
            "volume_true",
            "volume_approx",
            "membership_acc",
            "runtime_ms",
            "n_vertices",
            "status",
        ],
    )

    plot_metric_vs_n(results, "support_error", "Support function error", "Support error vs n", output_dir)
    plot_metric_vs_n(results, "volume_ratio", "Volume ratio (approx/true)", "Volume ratio vs n", output_dir)
    plot_metric_vs_n(results, "runtime_ms", "Runtime (ms)", "Runtime vs n", output_dir, logy=True)
    for dim, payload in plot_samples.items():
        plot_low_dim_visualisations(*payload, output_path=os.path.join(output_dir, f"vis_d{dim}.png"))
    return results


def run_high_dim_experiments(
    rng: np.random.Generator,
    output_dir: str,
    deephull_kwargs: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dims = [50, 100, 200, 500]
    ns = [2000, 5000, 10000]
    runtime_rows: List[Dict[str, Any]] = []
    support_rows: List[Dict[str, Any]] = []
    feasibility: Dict[Tuple[str, int, int], bool] = {}

    for dim in dims:
        for n in ns:
            for dataset_name in ["gaussian", "uniform_cube"]:
                points = DATASET_GENERATORS[dataset_name](rng, n, dim)
                hull_result, status = try_exact_hull_with_timeout(points, timeout_s=5.0)
                if status != "ok":
                    warnings.warn(f"SciPy hull skipped for d={dim}, n={n}, dataset={dataset_name}: {status}")

                method_outputs: List[MethodResult] = []
                method_outputs.append(run_extents(points, random_samples=4096))
                method_outputs.append(run_mvee(points, m=8, kappa=45))
                for method in ["original", "convex"]:
                    method_outputs.append(run_deephull(points, method=method, **deephull_kwargs))

                # Support function values for pairwise distances
                dirs = sample_directions(rng, 256, dim)
                support_by_method: Dict[str, np.ndarray] = {}

                for mres in method_outputs:
                    runtime_rows.append(
                        {
                            "dimension": dim,
                            "n_points": n,
                            "dataset": dataset_name,
                            "method": mres.method,
                            "runtime_ms": mres.runtime_ms,
                            "memory_bytes": estimate_memory_bytes(mres.vertices),
                            "status": mres.status,
                            "qhull_status": status,
                        }
                    )
                    if mres.status == "missing":
                        continue
                    support_vals = support_function(mres.vertices, dirs)
                    support_by_method[mres.method] = support_vals
                    feasibility[(mres.method, dim, n)] = mres.runtime_ms < 60000.0

                support_rows.extend(
                    [
                        {
                            "dimension": dim,
                            "n_points": n,
                            "dataset": dataset_name,
                            **row,
                        }
                        for row in pairwise_support_distribution(dirs, support_by_method)
                    ]
                )

    ensure_dir(output_dir)
    save_csv(
        os.path.join(output_dir, "high_dim_runtime.csv"),
        runtime_rows,
        fieldnames=["dimension", "n_points", "dataset", "method", "runtime_ms", "memory_bytes", "status", "qhull_status"],
    )
    save_csv(
        os.path.join(output_dir, "high_dim_support_pairs.csv"),
        support_rows,
        fieldnames=["dimension", "n_points", "dataset", "pair", "direction", "support_diff"],
    )

    plot_metric_vs_n(runtime_rows, "runtime_ms", "Runtime (ms)", "Runtime vs n (high-d)", output_dir, logy=True)
    plot_runtime_vs_dimension(runtime_rows, output_dir)
    plot_violin_support(
        support_rows,
        output_path=os.path.join(output_dir, "support_pair_violin.png"),
    )
    plot_feasibility_heatmap(
        feasibility,
        output_path=os.path.join(output_dir, "feasibility_heatmap.png"),
    )
    return runtime_rows, support_rows


def run_anomaly_detection(
    rng: np.random.Generator,
    output_dir: str,
    deephull_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    n_train = 10000
    n_ood = 5000
    dim = 10
    mean_id = np.zeros(dim)
    mean_ood = np.ones(dim) * 3.0
    cov = np.eye(dim)

    train_points = rng.multivariate_normal(mean_id, cov, size=n_train)
    ood_points = rng.multivariate_normal(mean_ood, cov, size=n_ood)
    test_points = np.vstack([train_points, ood_points])
    labels = np.concatenate([np.zeros(n_train, dtype=int), np.ones(n_ood, dtype=int)])

    method_outputs: List[MethodResult] = []
    method_outputs.append(run_extents(train_points, random_samples=4096))
    method_outputs.append(run_mvee(train_points, m=8, kappa=45))
    for method in ["original", "convex"]:
        method_outputs.append(run_deephull(train_points, method=method, **deephull_kwargs))

    def score_points(vertices: np.ndarray, pts: np.ndarray) -> np.ndarray:
        if vertices.size == 0:
            return np.zeros(pts.shape[0])
        directions = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-9)
        support_vals = support_function(vertices, directions)
        margins = support_vals - np.sum(directions * pts, axis=1)
        return -margins  # higher means more anomalous

    rows: List[Dict[str, Any]] = []
    roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    pr_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for mres in method_outputs:
        if mres.status == "missing":
            continue
        scores = score_points(mres.vertices, test_points)
        labels_np = labels.astype(int)
        auroc = _binary_auc(labels_np, scores)
        aupr = _average_precision(labels_np, scores)
        fpr, tpr = _roc_curve(labels_np, scores)
        recall, precision = _pr_curve(labels_np, scores)
        roc_curves[mres.method] = (fpr, tpr)
        pr_curves[mres.method] = (recall, precision)
        rows.append(
            {
                "method": mres.method,
                "auroc": auroc,
                "aupr": aupr,
                "runtime_ms": mres.runtime_ms,
                "n_vertices": int(mres.vertices.shape[0]),
            }
        )

    ensure_dir(output_dir)
    save_csv(
        os.path.join(output_dir, "anomaly_detection.csv"),
        rows,
        fieldnames=["method", "auroc", "aupr", "runtime_ms", "n_vertices"],
    )
    plot_anomaly_curves(roc_curves, pr_curves, output_dir)
    metrics_dict = {row["method"]: (row["auroc"], row["aupr"]) for row in rows}
    plot_bar_metrics(metrics_dict, os.path.join(output_dir, "anomaly_bar.png"))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run convex-hull experiments end-to-end.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Root directory for results and plots.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--deephull-epochs", type=int, default=150, help="Training epochs for DeepHull.")
    parser.add_argument("--deephull-lambda", type=float, default=2.0, help="Negative sample weight for DeepHull.")
    parser.add_argument("--deephull-epsilon", type=float, default=0.05, help="Boundary tolerance for DeepHull.")
    parser.add_argument("--deephull-lipschitz", type=float, default=1.0, help="Lipschitz constant for convex ICNN.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    deephull_kwargs = {
        "max_epochs": args.deephull_epochs,
        "lambda_neg": args.deephull_lambda,
        "level_set_epsilon": args.deephull_epsilon,
        "lipschitz_constant": args.deephull_lipschitz,
    }

    low_dir = os.path.join(args.output_dir, "low_dim")
    high_dir = os.path.join(args.output_dir, "high_dim")
    anomaly_dir = os.path.join(args.output_dir, "anomaly")

    print("[1/3] Running low-dimensional exact regime...")
    low_results = run_low_dim_experiments(rng, low_dir, deephull_kwargs)
    print(f"Saved low-dimensional results to {low_dir}")

    print("[2/3] Running high-dimensional scalability...")
    high_runtime, high_support = run_high_dim_experiments(rng, high_dir, deephull_kwargs)
    print(f"Saved high-dimensional results to {high_dir}")

    print("[3/3] Running anomaly-detection benchmark...")
    anomaly_results = run_anomaly_detection(rng, anomaly_dir, deephull_kwargs)
    print(f"Saved anomaly-detection results to {anomaly_dir}")

    # Aggregate index of produced files for convenience
    manifest = {
        "low_dim": [os.path.join(low_dir, fname) for fname in os.listdir(low_dir)],
        "high_dim": [os.path.join(high_dir, fname) for fname in os.listdir(high_dir)],
        "anomaly": [os.path.join(anomaly_dir, fname) for fname in os.listdir(anomaly_dir)],
    }
    save_json(os.path.join(args.output_dir, "manifest.json"), manifest)
    print(f"Experiment manifest written to {os.path.join(args.output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
