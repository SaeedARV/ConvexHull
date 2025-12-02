from typing import Optional

import numpy as np
import torch
from common import (
    vMF,
    householder_matrix,
)
from Ellipsoid import Ellipsoid
from common import sample_input

np.random.seed(41)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
torch.manual_seed(41)


def _select_device(requested: Optional[str] = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ConvexHullviaMVEE:
    """
    Class to compute the convex hull using the Minimum Volume Enclosing Ellipsoid (MVEE).
    """

    def __init__(self, points, device: Optional[str] = None):
        self.points = np.array(points)
        self.device = _select_device(device)
        self.dtype = torch.float32 if self.device.type in {"cuda", "mps"} else torch.float64
        self.points_tensor = torch.as_tensor(self.points, device=self.device, dtype=self.dtype)

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
        P = self.points_tensor
        n, d = P.shape
        # unit basis e1 = [1,0,...,0]
        e1 = torch.zeros(d, device=self.device, dtype=self.dtype)
        e1[0] = 1.0

        extents = dict([[tuple(x), []] for x in self.points])
        perp_vec = dict([[tuple(x), []] for x in self.points])
        rotated_vecs = dict([[tuple(x), []] for x in self.points])

        S_indices = set()
        U_t = torch.as_tensor(U, device=self.device, dtype=self.dtype)

        # for each original point
        for i in range(n):
            p = P[i]

            # 1) find the closest MVEE point
            closest_ellipsoid_vector = torch.as_tensor(E.project(p), device=self.device, dtype=self.dtype)
            p_hat = torch.as_tensor(E.normal_vector(closest_ellipsoid_vector), device=self.device, dtype=self.dtype)

            # 2) build the rotation/reflection sending e1 → p_hat
            R = householder_matrix(e1.cpu().numpy(), p_hat.cpu().numpy())
            R_t = torch.as_tensor(R, device=self.device, dtype=self.dtype)

            # 3) rotate all directions in U
            U_rot = U_t @ R_t.t()  # still shape (m,d)
            rotated_vecs[tuple(self.points[i])] = U_rot.detach().cpu().numpy()
            perp_vec[tuple(self.points[i])] = p_hat.detach().cpu().numpy()
            # 4) for each rotated direction, pick the supporting point in P
            #    (i.e. max dot with that direction)
            for u in U_rot:
                dots = torch.mv(P, u)
                s_idx = int(torch.argmax(dots).item())
                extents[tuple(self.points[s_idx])].append(u.detach().cpu().numpy())
                S_indices.add(s_idx)

        # assemble S as the unique selected points
        S = self.points[list(S_indices), :]
        if return_extents == True:
            return S, extents, rotated_vecs, perp_vec
        return S

    def compute(self, m=3, kappa=5, return_extents=False):
        """
        Computes the convex hull using MVEE and returns the hull vertices.
        """
        n, d = self.points.shape
        U = vMF(d, kappa).sample(m)
        E = Ellipsoid(self.points, device=self.device)
        S, extents, r, p = self.extents_estimation(U, E, return_extents)
        if return_extents == True:
            return S, extents, U, r, p
        return S


if __name__ == "__main__":
    Z = sample_input()
    conv = ConvexHullviaMVEE(Z)
    hull = conv.compute()
    from plots import plots
    from scipy.spatial import ConvexHull

    plots(Z, ConvexHull(Z)).all(hull)
