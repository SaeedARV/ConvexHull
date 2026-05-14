from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass
class SupportGapStats:
    max_gap: float
    mean_gap: float
    frac_gap_gt_eps: float


def argmax_dot_blocked(points: np.ndarray, directions: np.ndarray, *, block_size: int = 8192) -> np.ndarray:
    n_dirs = directions.shape[0]
    winners = np.empty(n_dirs, dtype=int)
    for start in range(0, n_dirs, block_size):
        end = min(start + block_size, n_dirs)
        dots = points @ directions[start:end].T
        winners[start:end] = np.argmax(dots, axis=0)
    return winners


def argmax_dot_with_ties(points: np.ndarray, directions: np.ndarray, *, atol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    dots = points @ directions.T
    max_vals = np.max(dots, axis=0)
    is_max = np.isclose(dots, max_vals, atol=atol)
    indices = [np.flatnonzero(is_max[:, j]) for j in range(directions.shape[0])]
    return max_vals, indices


def gather_unique(indices: Iterable[int]) -> np.ndarray:
    return np.unique(np.fromiter(indices, dtype=int))


def estimate_support_gap(
    points: np.ndarray,
    subset_points: np.ndarray,
    directions: np.ndarray,
    *,
    epsilon: float,
) -> SupportGapStats:
    omega_p = np.max(points @ directions.T, axis=0)
    omega_s = np.max(subset_points @ directions.T, axis=0)
    gaps = omega_p - omega_s
    return SupportGapStats(
        max_gap=float(np.max(gaps)),
        mean_gap=float(np.mean(gaps)),
        frac_gap_gt_eps=float(np.mean(gaps > epsilon)),
    )
