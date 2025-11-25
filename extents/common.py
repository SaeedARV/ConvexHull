import numpy as np
import cvxpy as cp
import numpy.linalg as la

np.random.seed(41)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

from scipy.stats import vonmises_fisher


def normalize(v):
    """
    Normalize a vector to unit length.
    """
    return v / np.linalg.norm(v)


class vMF:
    """
    Wrapper class for sampling from the von Mises-Fisher distribution using scipy.
    """

    def __init__(self, d, kappa, mu=None):
        """
        :param d: Dimension of the ambient space (samples lie on S^{d-1} ⊂ R^d)
        :param kappa: Concentration parameter (kappa = 0 is uniform on sphere)
        :param mu: Mean direction (unit vector); defaults to [1, 0, ..., 0] if not provided
        """
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
        """
        Draw m samples from the von Mises-Fisher distribution.
        :param m: Number of samples
        :return: (m x d) array of unit vectors
        """
        return self._dist.rvs(size=m)


def householder_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the Householder reflection H (d×d) such that H @ a = b,
    where a and b are assumed to be unit vectors in R^d.
    """
    v = a - b
    v = v / np.linalg.norm(v)
    H = np.eye(a.shape[0]) - 2.0 * np.outer(v, v)
    return H


def sample_input():
    X = [
        [-0.13536, 0.052424],
        [0.12526, -0.4626],
        [0.28357, -0.52009],
        [-0.07684, 0.39493],
        [-0.61311, -0.474],
        [-0.28483, -0.48858],
        [-0.38532, -0.016856],
        [-0.51643, 0.57121],
        [-0.30489, 0.73471],
        [0.74634, 0.35356],
        [-0.92925, -0.68531],
        [-0.16505, -0.75764],
        [0.60003, -0.91131],
        [0.13469, -0.22321],
        [0.55716, -0.6904],
        [0.50771, 0.11204],
        [-0.32228, 0.33077],
        [0.64648, -0.44766],
        [-0.28416, -1.0558],
        [-0.40915, -0.48119],
        [0.06225, 0.054254],
        [-0.21965, -0.35678],
        [0.46709, 0.029328],
        [0.80486, 0.42995],
        [-0.49260, -0.47918],
        [0.22455, -0.47123],
        [0.07945, 0.19404],
        [0.21867, 0.20911],
        [-0.36609, -0.71414],
        [-1.00461, -0.11672],
        [0.90198, -0.97434],
        [0.68393, -0.92937],
        [-0.61698, -0.25378],
        [0.70359, -0.47055],
        [0.43675, 0.56755],
        [0.58299, 0.024607],
        [0.25547, 0.31565],
        [0.44385, 0.028826],
        [-0.16477, -1.416],
        [-0.59128, -0.027423],
    ]
    return np.array(X)
