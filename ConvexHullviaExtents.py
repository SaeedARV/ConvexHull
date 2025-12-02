from typing import Optional

import numpy as np
import torch
from common import normalize

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


class ConvexHullviaExtents:
    """
    Class to compute and store the convex hull extents.
    """

    def __init__(self, points, device: Optional[str] = None):
        self.device = _select_device(device)
        self.dtype = torch.float32 if self.device.type in {"cuda", "mps"} else torch.float64
        self.points = np.array(points)
        self.X = np.array(points)
        self.X_tensor = torch.as_tensor(self.X, device=self.device, dtype=self.dtype)
        self.extents, self.random_extents, self.extents_max = None, None, None

    def get_extents(self, t=1000, return_max_extent=False):
        """
        Calculates the directions of the convex hull's extents in any dimension.

        For 2D, uses uniform angle sampling. For higher dimensions,
        uses uniform sampling on the unit sphere.

        Parameters:
        -----------
        t : int, optional
            Number of directions to sample. Default is 1000.
        return_max_extent : bool, optional
            If True, also returns the maximum extent direction for each point.
            Default is False.

        Returns:
        --------
        extents : dict
            Dictionary mapping each point to a list of directions where
            it is the supporting point.
        extents_max : dict, optional
            Dictionary mapping each point to its maximum extent direction.
            Only returned if return_max_extent is True.
        """
        _, d = self.X.shape

        # For 2D, use uniform angle sampling for better coverage
        # For higher dimensions, use uniform sampling on the unit sphere
        if d == 2:
            thetas = torch.linspace(0, 2 * np.pi, t, device=self.device, dtype=self.dtype)
            directions = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)
        else:
            # Sample uniformly on the unit sphere in d dimensions
            # by sampling from standard normal and normalizing
            directions = torch.randn(size=(t, d), device=self.device, dtype=self.dtype)
            norms = torch.norm(directions, dim=1, keepdim=True).clamp(min=1e-12)
            directions = directions / norms

        extents = dict([[tuple(x), []] for x in self.X])

        dot_products = self.X_tensor @ directions.T
        max_indices = torch.argmax(dot_products, dim=0)
        for i, direction in enumerate(directions):
            x = self.X[max_indices[i].item()]
            extents[tuple(x)].append(direction.detach().cpu().numpy())
        if return_max_extent:
            extents_max = dict([[tuple(x), []] for x in self.X])
            per_point = torch.argmax(dot_products, dim=1)
            for i, x in enumerate(self.X):
                dir_vec = directions[per_point[i]].detach().cpu().numpy()
                extents_max[tuple(x)] = normalize(dir_vec)

            return extents, extents_max
        return extents

    def get_random_extents(self, t, return_approx_hull=False):
        """
        Calculates the randomized directional extent of the convex hull's extents
        in any dimension.

        Parameters:
        -----------
        t : int
            Number of random directions to sample.
        return_approx_hull : bool, optional
            If True, also returns the set of points that have at least one extent.
            Default is False.

        Returns:
        --------
        random_extents : dict
            Dictionary mapping each point to a list of random directions where
            it is the supporting point.
        random_extents_set : set, optional
            Set of points that have at least one extent direction.
            Only returned if return_approx_hull is True.
        """
        _, d = self.X.shape
        random_directions = torch.randn(size=(t, d), device=self.device, dtype=self.dtype)
        norms = torch.norm(random_directions, dim=1, keepdim=True).clamp(min=1e-12)
        random_directions = random_directions / norms
        random_extents = dict({})
        for x in self.X:
            random_extents[tuple(x)] = []
        random_dot_products = self.X_tensor @ random_directions.T
        max_indices = torch.argmax(random_dot_products, dim=0)
        for i, random_direction in enumerate(random_directions):
            x = self.X[max_indices[i].item()]
            random_extents[tuple(x)].append(random_direction.detach().cpu().numpy())

        if return_approx_hull:
            random_extents_set = set()
            for x in self.X:
                if len(random_extents[tuple(x)]) > 0:
                    random_extents_set.add(tuple(x))
            return random_extents, random_extents_set
        else:
            return random_extents
