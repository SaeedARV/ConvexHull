"""von Mises-Fisher sampling utilities."""

from __future__ import annotations

import numpy as np

from convex_hull.approximation._random import get_rng
from convex_hull.approximation._sampling import sample_vmf


class vMF:
    """Sampler for the von Mises-Fisher distribution on the unit sphere."""

    def __init__(self, d, kappa, mu=None, random_state=None):
        self.d = int(d)
        self.kappa = float(kappa)
        if mu is None:
            mu = np.zeros(self.d)
            mu[0] = 1.0
        mu = np.asarray(mu, dtype=float)
        norm = np.linalg.norm(mu)
        if norm == 0.0:
            raise ValueError("mu must be nonzero")
        self.mu = mu / norm
        self.random_state = random_state

    def sample(self, m):
        """Draw m samples as an m x d array of unit vectors."""
        rng = get_rng(self.random_state)
        return sample_vmf(int(m), self.d, self.mu, self.kappa, rng)
