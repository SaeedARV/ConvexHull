import numpy as np

from convex_hull.geometry import Ellipsoid
from convex_hull.sampling import vMF
from convex_hull.utils import householder_matrix, sample_input

np.random.seed(41)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


class ConvexHullviaMVEE:
    """
    Class to compute the convex hull using the Minimum Volume Enclosing Ellipsoid (MVEE).
    """

    def __init__(self, points):
        self.points = np.array(points)

    def extents_estimation(self, U, E, return_extents=True):
        """
        Implements the “Extents Estimation” subroutine.
        Inputs:
        P : (n,d) array of input points in R^d
        U : (m,d) array of sampled directions on S^{d-1}
        E : (k,d) array of extremal points from MVEE(P)
        Returns:
        S : an array of the selected subset of P (shape ≤ n×d)
        """
        P = self.points
        n, d = P.shape
        # unit basis e1 = [1,0,...,0]
        e1 = np.zeros(d)
        e1[0] = 1.0

        extents = dict([[tuple(x), []] for x in P])
        perp_vec = dict([[tuple(x), []] for x in P])
        rotated_vecs = dict([[tuple(x), []] for x in P])

        S_indices = set()

        # for each original point
        for i in range(n):
            p = P[i]

            # 1) find the closest MVEE point
            closest_ellipsoid_vector = E.project(p)
            p_hat = E.normal_vector(closest_ellipsoid_vector)

            # 2) build the rotation/reflection sending e1 → p_hat
            R = householder_matrix(e1, p_hat)

            # 3) rotate all directions in U
            U_rot = U @ R.T  # still shape (m,d)
            rotated_vecs[tuple(p)] = U_rot
            perp_vec[tuple(p)] = p_hat
            # 4) for each rotated direction, pick the supporting point in P
            #    (i.e. max dot with that direction)
            for u in U_rot:
                # for u in [p_hat]:
                # compute dot products
                dots = P.dot(u)
                s_idx = int(np.argmax(dots))
                extents[tuple(P[s_idx])].append(u)
                S_indices.add(s_idx)

        # assemble S as the unique selected points
        S = P[list(S_indices), :]
        if return_extents == True:
            return S, extents, rotated_vecs, perp_vec
        return S

    def compute(self, m=3, kappa=5, return_extents=False):
        """
        Computes the convex hull using MVEE and returns the hull vertices.
        """
        n, d = self.points.shape
        U = vMF(d, kappa).sample(m)
        E = Ellipsoid(self.points)
        S, extents, r, p = self.extents_estimation(U, E, return_extents)
        if return_extents == True:
            return S, extents, U, r, p
        return S


if __name__ == "__main__":
    Z = sample_input()
    conv = ConvexHullviaMVEE(Z)
    hull = conv.compute()
    from convex_hull.visualization import plots
    from scipy.spatial import ConvexHull

    plots(Z, ConvexHull(Z)).all(hull)
