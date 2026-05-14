"""von Mises-Fisher sampling utilities."""

import numpy as np
from scipy.stats import vonmises_fisher


class vMF:
    """Wrapper for sampling from the von Mises-Fisher distribution."""

    def __init__(self, d, kappa, mu=None):
        self.d = d
        self.kappa = kappa
        if mu is None:
            mu = np.zeros(d)
            mu[0] = 1.0
        else:
            mu = np.asarray(mu, dtype=float)
            mu /= np.linalg.norm(mu)
        self.mu = mu
        self._dist = vonmises_fisher(mu=self.mu, kappa=self.kappa)

    def sample(self, m):
        """Draw m samples as an m x d array of unit vectors."""
        return self._dist.rvs(size=m)
