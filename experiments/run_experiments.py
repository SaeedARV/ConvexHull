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
import gc
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

# Non-interactive backend so plots can be written without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import ConvexHull, QhullError  # noqa: E402

from extents.convexhull import ConvexHullviaExtents  # noqa: E402
from ConvexHullviaMVEE import ConvexHullviaMVEE  # noqa: E402

try:
    from ConvexHullviaDeepHull import ConvexHullviaDeepHull  # type: ignore
    import torch  # type: ignore
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


def append_csv_rows(path: str, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    import csv

    ensure_dir(os.path.dirname(path))
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def load_csv_rows(
    path: str,
    int_fields: Sequence[str] = (),
    float_fields: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    import csv

    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, Any] = dict(row)
            valid = True
            for field in int_fields:
                value = parsed.get(field, "")
                if value == "":
                    valid = False
                    break
                try:
                    parsed[field] = int(float(value))
                except ValueError:
                    valid = False
                    break
            if not valid:
                continue
            for field in float_fields:
                value = parsed.get(field, "")
                if value == "":
                    parsed[field] = math.nan
                else:
                    try:
                        parsed[field] = float(value)
                    except ValueError:
                        parsed[field] = math.nan
            rows.append(parsed)
    return rows


def build_key_set(rows: List[Dict[str, Any]], key_fields: Sequence[str]) -> set[Tuple[Any, ...]]:
    return {tuple(row[field] for field in key_fields) for row in rows}


def log(message: str) -> None:
    print(message, flush=True)


def _run_method_worker(
    method_name: str,
    points: np.ndarray,
    method_params: Dict[str, Any],
    deephull_kwargs: Dict[str, Any],
) -> MethodResult:
    if method_name == "Extents":
        return run_extents(points, **method_params)
    if method_name == "MVEE":
        return run_mvee(points, **method_params)
    if method_name.startswith("DeepHull-"):
        method = method_name.split("-", 1)[1]
        order_indices = method_params.get("order_indices", True)
        return run_deephull(points, method=method, order_indices=order_indices, **deephull_kwargs)
    raise ValueError(f"Unknown method name: {method_name}")


def run_method(
    method_name: str,
    points: np.ndarray,
    method_params: Dict[str, Any],
    deephull_kwargs: Dict[str, Any],
) -> MethodResult:
    return _run_method_worker(method_name, points, method_params, deephull_kwargs)


def run_and_log(
    method_name: str,
    points: np.ndarray,
    method_params: Dict[str, Any],
    deephull_kwargs: Dict[str, Any],
) -> MethodResult:
    log(f"  - {method_name}: starting")
    result = run_method(method_name, points, method_params, deephull_kwargs)
    log(f"  - {method_name}: finished ({result.status}, {result.runtime_ms:.1f} ms)")
    cleanup_after_method()
    return result


def compute_low_dim_metrics_safe(
    rng: np.random.Generator,
    points: np.ndarray,
    vertices: np.ndarray,
) -> Dict[str, Any]:
    try:
        seed = int(rng.integers(0, 2**31 - 1))
        local_rng = np.random.default_rng(seed)
        true_hull = ConvexHull(points)
        return compute_low_dim_metrics(local_rng, points, true_hull, MethodResult("tmp", vertices, [], 0.0))
    except Exception as exc:
        log(f"  - metrics error {exc!r}")
        return {
            "support_error": math.nan,
            "volume_ratio": math.nan,
            "volume_true": math.nan,
            "volume_approx": math.nan,
            "membership_acc": math.nan,
            "runtime_ms": math.nan,
            "n_vertices": int(vertices.shape[0]),
            "status": "metric_error",
        }


def cleanup_after_method() -> None:
    gc.collect()
    if ConvexHullviaDeepHull is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


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


# ---------------------------------------------------------------------------
# Solver wrappers
# ---------------------------------------------------------------------------
def run_extents(
    points: np.ndarray,
    random_samples: int = 2048,
    order_indices: bool = True,
) -> MethodResult:
    start = time.perf_counter()
    solver = ConvexHullviaExtents(points)
    _, approx_set = solver.get_random_extents(
        random_samples,
        return_approx_hull=True,
        method="auto",
    )
    runtime_ms = (time.perf_counter() - start) * 1000.0
    if not approx_set:
        return MethodResult("Extents", np.empty((0, points.shape[1])), [], runtime_ms, status="empty")
    approx_vertices = np.array(list(approx_set))
    try:
        indices = _points_to_indices(points, approx_vertices)
        if order_indices:
            indices = _order_indices(points, indices)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"Extents index mapping failed: {exc}")
        indices = []
    return MethodResult("Extents", approx_vertices, indices, runtime_ms)


def run_mvee(points: np.ndarray, order_indices: bool = True, **kwargs: Any) -> MethodResult:
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
        if order_indices:
            indices = _order_indices(points, indices)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"MVEE index mapping failed: {exc}")
        indices = []
    return MethodResult("MVEE", approx_vertices, indices, runtime_ms)


def run_deephull(
    points: np.ndarray,
    method: str,
    order_indices: bool = True,
    **kwargs: Any,
) -> MethodResult:
    start = time.perf_counter()
    if ConvexHullviaDeepHull is None:
        return MethodResult(f"DeepHull-{method}", np.empty((0, points.shape[1])), [], 0.0, status="missing")
    model = ConvexHullviaDeepHull(method=method, **kwargs)
    indices = model.predict_indices(points)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    approx_vertices = points[indices] if indices else np.empty((0, points.shape[1]))
    indices = _sanitize_indices(indices, points.shape[0])
    if order_indices:
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
    resume: bool = False,
) -> List[Dict[str, Any]]:
    dims = [2, 3, 5, 8]
    ns = [200, 500, 1000]
    results_path = os.path.join(output_dir, "low_dim_results.csv")
    fieldnames = [
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
    ]
    existing_rows = (
        load_csv_rows(
            results_path,
            int_fields=["dimension", "n_points", "n_vertices"],
            float_fields=[
                "support_error",
                "volume_ratio",
                "volume_true",
                "volume_approx",
                "membership_acc",
                "runtime_ms",
            ],
        )
        if resume
        else []
    )
    if not resume:
        ensure_dir(output_dir)
        if os.path.exists(results_path):
            with open(results_path, "w", encoding="utf-8"):
                pass
    existing_keys = build_key_set(existing_rows, ("dimension", "n_points", "dataset", "method")) if resume else set()

    plot_samples: Dict[int, Tuple[np.ndarray, ConvexHull, Dict[str, np.ndarray]]] = {}
    method_names = ["Extents", "MVEE"]
    if ConvexHullviaDeepHull is not None:
        method_names.extend(["DeepHull-original", "DeepHull-convex"])

    for dim in dims:
        for n in ns:
            if dim == 8 and n >= 500:
                log(f"[low] skip d=8 n={n} (excluded due to memory constraints)")
                continue
            for dataset_name in ["uniform_cube", "gaussian"]:
                missing_methods = [
                    method for method in method_names if (dim, n, dataset_name, method) not in existing_keys
                ]
                need_visual = dim in (2, 3) and dim not in plot_samples
                skip_dataset = resume and not missing_methods and not need_visual
                if skip_dataset:
                    log(f"[low] skip d={dim} n={n} dataset={dataset_name} (already in {results_path})")
                else:
                    log(f"[low] start d={dim} n={n} dataset={dataset_name}")
                    points = DATASET_GENERATORS[dataset_name](rng, n, dim)
                    true_hull = ConvexHull(points)
                    methods_to_run = method_names if need_visual else missing_methods
                    approximations: Dict[str, np.ndarray] = {}
                    for method_name in method_names:
                        if method_name not in methods_to_run:
                            continue
                        if method_name == "Extents":
                            mres = run_and_log(
                                "Extents",
                                points,
                                {"random_samples": 2048},
                                deephull_kwargs,
                            )
                        elif method_name == "MVEE":
                            mres = run_and_log(
                                "MVEE",
                                points,
                                {"m": 8, "kappa": 45},
                                deephull_kwargs,
                            )
                        else:
                            method = method_name.split("-", 1)[1]
                            mres = run_and_log(method_name, points, {}, deephull_kwargs)
                        if mres.vertices.size and need_visual:
                            approximations[mres.method] = mres.vertices
                        if mres.status == "missing":
                            continue
                        key = (dim, n, dataset_name, mres.method)
                        if key in existing_keys:
                            continue
                        metrics = compute_low_dim_metrics_safe(rng, points, mres.vertices)
                        row = {
                            "dimension": dim,
                            "n_points": n,
                            "dataset": dataset_name,
                            "method": mres.method,
                            **metrics,
                        }
                        existing_keys.add(key)
                        append_csv_rows(results_path, [row], fieldnames)
                        log(f"    saved {mres.method} row to {results_path}")

                if not skip_dataset:
                    # Store samples for visualisation
                    if dim in (2, 3) and (dim not in plot_samples):
                        plot_samples[dim] = (points, true_hull, approximations)

            for cond in [1, 10, 100]:
                ani_dataset = f"anisotropic_cond{cond}"
                missing_ani = [method for method in method_names if (dim, n, ani_dataset, method) not in existing_keys]
                if resume and not missing_ani:
                    log(f"[low] skip d={dim} n={n} dataset={ani_dataset} (already in {results_path})")
                    continue
                if not missing_ani:
                    log(f"[low] skip d={dim} n={n} dataset={ani_dataset} (already computed)")
                    continue
                log(f"[low] start d={dim} n={n} dataset={ani_dataset}")
                ani_points = generate_anisotropic_gaussian(rng, n, dim, condition=cond)
                ani_hull = ConvexHull(ani_points)
                for method_name in method_names:
                    if method_name not in missing_ani:
                        continue
                    if method_name == "Extents":
                        mres = run_and_log(
                            "Extents",
                            ani_points,
                            {"random_samples": 2048},
                            deephull_kwargs,
                        )
                    elif method_name == "MVEE":
                        mres = run_and_log(
                            "MVEE",
                            ani_points,
                            {"m": 8, "kappa": 45},
                            deephull_kwargs,
                        )
                    else:
                        method = method_name.split("-", 1)[1]
                        mres = run_and_log(method_name, ani_points, {}, deephull_kwargs)
                    if mres.status == "missing":
                        continue
                    metrics = compute_low_dim_metrics_safe(rng, ani_points, mres.vertices)
                    row = {
                        "dimension": dim,
                        "n_points": n,
                        "dataset": ani_dataset,
                        "method": mres.method,
                        **metrics,
                    }
                    key = (dim, n, ani_dataset, mres.method)
                    if key in existing_keys:
                        continue
                    existing_keys.add(key)
                    append_csv_rows(results_path, [row], fieldnames)
                    log(f"    saved {mres.method} row to {results_path}")

    plot_rows = load_csv_rows(
        results_path,
        int_fields=["dimension", "n_points", "n_vertices"],
        float_fields=[
            "support_error",
            "volume_ratio",
            "volume_true",
            "volume_approx",
            "membership_acc",
            "runtime_ms",
        ],
    )
    if not plot_rows:
        log("[low] no results available; skipping plots")
        return plot_rows
    ensure_dir(output_dir)
    plot_metric_vs_n(plot_rows, "support_error", "Support function error", "Support error vs n", output_dir)
    plot_metric_vs_n(plot_rows, "volume_ratio", "Volume ratio (approx/true)", "Volume ratio vs n", output_dir)
    plot_metric_vs_n(plot_rows, "runtime_ms", "Runtime (ms)", "Runtime vs n", output_dir, logy=True)
    for dim, payload in plot_samples.items():
        plot_low_dim_visualisations(*payload, output_path=os.path.join(output_dir, f"vis_d{dim}.png"))
    return plot_rows


def run_high_dim_experiments(
    rng: np.random.Generator,
    output_dir: str,
    deephull_kwargs: Dict[str, Any],
    resume: bool = False,
    extents_samples: int = 2048,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dims = [50, 100, 200, 500]
    ns = [2000, 5000, 10000]
    runtime_path = os.path.join(output_dir, "high_dim_runtime.csv")
    support_path = os.path.join(output_dir, "high_dim_support_pairs.csv")
    runtime_fieldnames = [
        "dimension",
        "n_points",
        "dataset",
        "method",
        "runtime_ms",
        "memory_bytes",
        "status",
    ]
    support_fieldnames = ["dimension", "n_points", "dataset", "pair", "direction", "support_diff"]
    runtime_rows_existing = (
        load_csv_rows(
            runtime_path,
            int_fields=["dimension", "n_points"],
            float_fields=["runtime_ms", "memory_bytes"],
        )
        if resume
        else []
    )
    support_rows_existing = (
        load_csv_rows(
            support_path,
            int_fields=["dimension", "n_points", "direction"],
            float_fields=["support_diff"],
        )
        if resume
        else []
    )
    if not resume:
        ensure_dir(output_dir)
        if os.path.exists(runtime_path):
            with open(runtime_path, "w", encoding="utf-8"):
                pass
        if os.path.exists(support_path):
            with open(support_path, "w", encoding="utf-8"):
                pass
    runtime_keys = (
        build_key_set(runtime_rows_existing, ("dimension", "n_points", "dataset", "method")) if resume else set()
    )
    support_keys = (
        build_key_set(support_rows_existing, ("dimension", "n_points", "dataset", "pair", "direction")) if resume else set()
    )
    method_names = ["Extents", "MVEE"]
    if ConvexHullviaDeepHull is not None:
        method_names.extend(["DeepHull-original", "DeepHull-convex"])

    for dim in dims:
        for n in ns:
            for dataset_name in ["gaussian", "uniform_cube"]:
                expected_pairs = [
                    f"{m1} vs {m2}" for m1, m2 in itertools.combinations(method_names, 2)
                ]
                runtime_missing = any(
                    (dim, n, dataset_name, method) not in runtime_keys for method in method_names
                )
                support_missing = False
                if expected_pairs:
                    support_missing = any(
                        (dim, n, dataset_name, pair, direction) not in support_keys
                        for pair in expected_pairs
                        for direction in range(256)
                    )
                if resume and not runtime_missing and not support_missing:
                    log(f"[high] skip d={dim} n={n} dataset={dataset_name} (already in {output_dir})")
                    continue

                log(f"[high] start d={dim} n={n} dataset={dataset_name}")
                points = DATASET_GENERATORS[dataset_name](rng, n, dim)

                method_outputs: List[MethodResult] = []
                method_outputs.append(
                    run_and_log(
                        "Extents",
                        points,
                        {"random_samples": extents_samples, "order_indices": False},
                        deephull_kwargs,
                    )
                )
                method_outputs.append(
                    run_and_log(
                        "MVEE",
                        points,
                        {"m": 8, "kappa": 45, "order_indices": False},
                        deephull_kwargs,
                    )
                )
                for method in ["original", "convex"]:
                    if f"DeepHull-{method}" in method_names:
                        method_outputs.append(
                            run_and_log(
                                f"DeepHull-{method}",
                                points,
                                {"order_indices": False},
                                deephull_kwargs,
                            )
                        )

                # Support function values for pairwise distances
                dirs = sample_directions(rng, 256, dim)
                support_by_method: Dict[str, np.ndarray] = {}
                for mres in method_outputs:
                    row = {
                        "dimension": dim,
                        "n_points": n,
                        "dataset": dataset_name,
                        "method": mres.method,
                        "runtime_ms": mres.runtime_ms,
                        "memory_bytes": estimate_memory_bytes(mres.vertices),
                        "status": mres.status,
                    }
                    key = (dim, n, dataset_name, mres.method)
                    if key not in runtime_keys:
                        runtime_keys.add(key)
                        append_csv_rows(runtime_path, [row], runtime_fieldnames)
                        log(f"    saved {mres.method} row to {runtime_path}")
                    if mres.status == "missing":
                        continue
                    support_vals = support_function(mres.vertices, dirs)
                    support_by_method[mres.method] = support_vals

                support_rows_to_append: List[Dict[str, Any]] = []
                for row in pairwise_support_distribution(dirs, support_by_method):
                    entry = {
                        "dimension": dim,
                        "n_points": n,
                        "dataset": dataset_name,
                        **row,
                    }
                    key = (dim, n, dataset_name, entry["pair"], entry["direction"])
                    if key in support_keys:
                        continue
                    support_keys.add(key)
                    support_rows_to_append.append(entry)
                if support_rows_to_append:
                    append_csv_rows(support_path, support_rows_to_append, support_fieldnames)
                    log(f"[high] appended {len(support_rows_to_append)} rows to {support_path}")

    plot_runtime_rows = load_csv_rows(
        runtime_path,
        int_fields=["dimension", "n_points"],
        float_fields=["runtime_ms", "memory_bytes"],
    )
    plot_support_rows = load_csv_rows(
        support_path,
        int_fields=["dimension", "n_points", "direction"],
        float_fields=["support_diff"],
    )
    if not plot_runtime_rows:
        log("[high] no results available; skipping plots")
        return plot_runtime_rows, plot_support_rows
    feasibility: Dict[Tuple[str, int, int], bool] = {}
    for row in plot_runtime_rows:
        runtime = row.get("runtime_ms")
        if runtime is None or not np.isfinite(runtime):
            continue
        feasibility[(str(row["method"]), int(row["dimension"]), int(row["n_points"]))] = float(runtime) < 60000.0
    ensure_dir(output_dir)

    plot_metric_vs_n(plot_runtime_rows, "runtime_ms", "Runtime (ms)", "Runtime vs n (high-d)", output_dir, logy=True)
    plot_runtime_vs_dimension(plot_runtime_rows, output_dir)
    plot_violin_support(
        plot_support_rows,
        output_path=os.path.join(output_dir, "support_pair_violin.png"),
    )
    plot_feasibility_heatmap(
        feasibility,
        output_path=os.path.join(output_dir, "feasibility_heatmap.png"),
    )
    return plot_runtime_rows, plot_support_rows


def run_anomaly_detection(
    rng: np.random.Generator,
    output_dir: str,
    deephull_kwargs: Dict[str, Any],
    resume: bool = False,
) -> List[Dict[str, Any]]:
    results_path = os.path.join(output_dir, "anomaly_detection.csv")
    fieldnames = ["method", "auroc", "aupr", "runtime_ms", "n_vertices"]
    rows = (
        load_csv_rows(
            results_path,
            int_fields=["n_vertices"],
            float_fields=["auroc", "aupr", "runtime_ms"],
        )
        if resume
        else []
    )
    if not resume:
        ensure_dir(output_dir)
        if os.path.exists(results_path):
            with open(results_path, "w", encoding="utf-8"):
                pass
    existing_methods = {row["method"] for row in rows} if resume else set()
    method_names = ["Extents", "MVEE"]
    if ConvexHullviaDeepHull is not None:
        method_names.extend(["DeepHull-original", "DeepHull-convex"])
    plots_ready = all(
        os.path.exists(os.path.join(output_dir, fname))
        for fname in ["roc_curves.png", "pr_curves.png", "anomaly_bar.png"]
    )
    if resume and set(method_names).issubset(existing_methods) and plots_ready:
        log(f"[anomaly] skip (already in {results_path})")
        return rows

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
    log("[anomaly] start")
    method_outputs.append(
        run_and_log(
            "Extents",
            train_points,
            {"random_samples": 4096},
            deephull_kwargs,
        )
    )
    method_outputs.append(
        run_and_log(
            "MVEE",
            train_points,
            {"m": 8, "kappa": 45},
            deephull_kwargs,
        )
    )
    for method in ["original", "convex"]:
        if f"DeepHull-{method}" in method_names:
            method_outputs.append(
                run_and_log(
                    f"DeepHull-{method}",
                    train_points,
                    {},
                    deephull_kwargs,
                )
            )

    def score_points(vertices: np.ndarray, pts: np.ndarray) -> np.ndarray:
        if vertices.size == 0:
            return np.zeros(pts.shape[0])
        directions = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-9)
        support_vals = support_function(vertices, directions)
        margins = support_vals - np.sum(directions * pts, axis=1)
        return -margins  # higher means more anomalous

    roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    pr_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    log("[anomaly] scoring methods")
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
        row = {
            "method": mres.method,
            "auroc": auroc,
            "aupr": aupr,
            "runtime_ms": mres.runtime_ms,
            "n_vertices": int(mres.vertices.shape[0]),
        }
        if mres.method not in existing_methods:
            existing_methods.add(mres.method)
            rows.append(row)
            append_csv_rows(results_path, [row], fieldnames)
            log(f"    saved {mres.method} row to {results_path}")
        log(f"  - {mres.method}: {mres.status} ({mres.runtime_ms:.1f} ms)")

    ensure_dir(output_dir)
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
    parser.add_argument(
        "--steps",
        type=str,
        default="low,high,anomaly",
        help="Comma-separated list of steps to run: low, high, anomaly.",
    )
    parser.add_argument(
        "--high-extents-samples",
        type=int,
        default=2048,
        help="Number of random directions for Extents in high-dimensional runs.",
    )
    parser.add_argument("--deephull-epochs", type=int, default=150, help="Training epochs for DeepHull.")
    parser.add_argument("--deephull-lambda", type=float, default=2.0, help="Negative sample weight for DeepHull.")
    parser.add_argument("--deephull-epsilon", type=float, default=0.05, help="Boundary tolerance for DeepHull.")
    parser.add_argument("--deephull-lipschitz", type=float, default=1.0, help="Lipschitz constant for convex ICNN.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing CSVs and append new rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = {step.strip().lower() for step in args.steps.split(",") if step.strip()}
    valid_steps = {"low", "high", "anomaly"}
    unknown = steps - valid_steps
    if unknown:
        raise ValueError(f"Unknown steps: {', '.join(sorted(unknown))}. Use low, high, anomaly.")
    if not steps:
        raise ValueError("No steps selected. Use --steps low,high,anomaly (or a subset).")

    rng = np.random.default_rng(args.seed)
    deephull_kwargs = {
        "max_epochs": args.deephull_epochs,
        "lambda_neg": args.deephull_lambda,
        "level_set_epsilon": args.deephull_epsilon,
        "lipschitz_constant": args.deephull_lipschitz,
    }
    high_extents_samples = int(args.high_extents_samples)

    low_dir = os.path.join(args.output_dir, "low_dim")
    high_dir = os.path.join(args.output_dir, "high_dim")
    anomaly_dir = os.path.join(args.output_dir, "anomaly")

    if "low" in steps:
        print("[low] Running low-dimensional exact regime...")
        low_results = run_low_dim_experiments(rng, low_dir, deephull_kwargs, resume=args.resume)
        print(f"Saved low-dimensional results to {low_dir}")

    if "high" in steps:
        print("[high] Running high-dimensional scalability...")
        high_runtime, high_support = run_high_dim_experiments(
            rng,
            high_dir,
            deephull_kwargs,
            resume=args.resume,
            extents_samples=high_extents_samples,
        )
        print(f"Saved high-dimensional results to {high_dir}")

    if "anomaly" in steps:
        print("[anomaly] Running anomaly-detection benchmark...")
        anomaly_results = run_anomaly_detection(rng, anomaly_dir, deephull_kwargs, resume=args.resume)
        print(f"Saved anomaly-detection results to {anomaly_dir}")

    def list_dir(path: str) -> List[str]:
        if not os.path.isdir(path):
            return []
        return [os.path.join(path, fname) for fname in os.listdir(path)]

    # Aggregate index of produced files for convenience
    manifest = {
        "low_dim": list_dir(low_dir),
        "high_dim": list_dir(high_dir),
        "anomaly": list_dir(anomaly_dir),
    }
    save_json(os.path.join(args.output_dir, "manifest.json"), manifest)
    print(f"Experiment manifest written to {os.path.join(args.output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
