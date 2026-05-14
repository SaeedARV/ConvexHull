import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from convex_hull.utils import householder_matrix, sample_input
from convex_hull.visualization import plots

np.random.seed(41)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def sample_sphere(n, d=2):
    """
    Sample n points uniformly from the surface of a d-dimensional sphere.
    """
    points = np.random.normal(size=(n, d))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    A = np.array([[4, 0], [0, 1]])
    H = householder_matrix(np.array([1, 0]), np.array([-2.8, -0.75]))
    points = points @ A @ H.T
    return points


def sample_rotated(n, d=2):
    """
    Sample n points from a rotated d-dimensional normal distribution.
    """
    X = np.random.normal(loc=(0, 0), scale=(0.7, 1), size=(n, 2))
    H = householder_matrix(np.array([1, 0]), np.array([-2.8, -0.75]))
    X = X @ H.T
    return X


if __name__ == "__main__":
    n = 50
    n = 30

    """
    Create a set of random points in 2D.
    """
    X = sample_rotated(n)
    # X = sample_sphere(n)
    # X[1] = [-0.750, -1.430]
    # X = np.array([[1, 0], [-1, 0], [0, -1], [0, 0.9]])
    # print(X)
    # X = sample_input()
    """
    Plot the convex hull of the points.
    """
    hull = ConvexHull(X)
    plt = plots(X, hull)
    # x = plt.Plain()
    # x = plt.AllviaExtents()
    x = plt.AllviaMVEE()
    # x = plt.plot_MVEE()
    x.show()
