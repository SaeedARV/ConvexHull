from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def sample_uniform_sphere(m: int, d: int, rng: np.random.Generator) -> np.ndarray:
    if m <= 0 or d <= 0:
        raise ValueError("m and d must be positive")
    out = np.empty((m, d), dtype=float)
    filled = 0
    while filled < m:
        need = m - filled
        z = rng.standard_normal(size=(need, d))
        norms = np.linalg.norm(z, axis=1)
        mask = norms > 0
        if not np.any(mask):
            continue
        z = z[mask]
        norms = norms[mask]
        out[filled : filled + z.shape[0]] = z / norms[:, None]
        filled += z.shape[0]
    return out


def householder_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have same shape")
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    if np.allclose(a, b):
        return np.eye(a.shape[0])
    if np.allclose(a, -b):
        # pick a vector orthogonal to a
        idx = np.argmin(np.abs(a))
        v = np.zeros_like(a)
        v[idx] = 1.0
    else:
        v = a - b
    v = v / np.linalg.norm(v)
    return np.eye(a.shape[0]) - 2.0 * np.outer(v, v)


def householder_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    if np.allclose(a, b):
        return np.zeros_like(a)
    if np.allclose(a, -b):
        idx = np.argmin(np.abs(a))
        v = np.zeros_like(a)
        v[idx] = 1.0
    else:
        v = a - b
    return v / np.linalg.norm(v)


def apply_householder_to_rows(U: np.ndarray, v: np.ndarray) -> np.ndarray:
    if np.allclose(v, 0):
        return U.copy()
    proj = U @ v
    return U - 2.0 * proj[:, None] * v[None, :]


def _sample_vmf_w(d: int, kappa: float, rng: np.random.Generator) -> float:
    if kappa == 0.0:
        return rng.uniform(-1.0, 1.0)
    b = (-2.0 * kappa + math.sqrt(4.0 * kappa * kappa + (d - 1) ** 2)) / (d - 1)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (d - 1) * math.log(1.0 - x0 * x0)
    while True:
        z = rng.beta((d - 1) / 2.0, (d - 1) / 2.0)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = rng.uniform(0.0, 1.0)
        if kappa * w + (d - 1) * math.log(1.0 - x0 * w) - c >= math.log(u):
            return w


def sample_vmf_base(m: int, d: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    if m <= 0 or d <= 0:
        raise ValueError("m and d must be positive")
    if d == 1:
        samples = np.sign(rng.standard_normal(size=m))
        return samples[:, None]
    w = np.array([_sample_vmf_w(d, kappa, rng) for _ in range(m)], dtype=float)
    v = sample_uniform_sphere(m, d - 1, rng)
    factor = np.sqrt(1.0 - w * w)
    return np.concatenate([w[:, None], factor[:, None] * v], axis=1)


def sample_vmf(m: int, d: int, mu: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    base = sample_vmf_base(m, d, kappa, rng)
    e1 = np.zeros(d)
    e1[0] = 1.0
    v = householder_vector(e1, mu)
    return apply_householder_to_rows(base, v)
