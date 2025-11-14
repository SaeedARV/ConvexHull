import numpy as np
from common import normalize

np.random.seed(41)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


class ConvexHullviaExtents:
    """
    Class to compute and store the convex hull extents.
    """

    def __init__(self, points):
        self.points = np.array(points)
        self.X = np.array(points)
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
        n, d = self.X.shape

        # For 2D, use uniform angle sampling for better coverage
        # For higher dimensions, use uniform sampling on the unit sphere
        if d == 2:
            directions = np.array(
                [
                    np.array([np.cos(theta), np.sin(theta)])
                    for theta in np.linspace(0, 2 * np.pi, t)
                ]
            )
        else:
            # Sample uniformly on the unit sphere in d dimensions
            # by sampling from standard normal and normalizing
            directions = np.random.normal(size=(t, d))
            norms = np.linalg.norm(directions, axis=1)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            directions = directions / norms[:, np.newaxis]

        extents = dict([[tuple(x), []] for x in self.X])

        dot_products = np.dot(self.X, directions.T)
        for i, direction in enumerate(directions):
            x = self.X[np.argmax(dot_products[:, i])]
            extents[tuple(x)].append(direction)
        if return_max_extent:
            extents_max = dict([[tuple(x), []] for x in self.X])
            for i, x in enumerate(self.X):
                extents_max[tuple(x)] = normalize(
                    directions[np.argmax(dot_products[i, :])]
                )

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
        n, d = self.X.shape
        random_directions = np.random.normal(size=(t, d))
        norms = np.linalg.norm(random_directions, axis=1)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        random_directions = random_directions / norms[:, np.newaxis]
        random_extents = dict({})
        for x in self.X:
            random_extents[tuple(x)] = []
        random_dot_products = np.dot(self.X, random_directions.T)
        for i, random_direction in enumerate(random_directions):
            x = self.X[np.argmax(random_dot_products[:, i])]
            random_extents[tuple(x)].append(random_direction)

        if return_approx_hull:
            random_extents_set = set()
            for x in self.X:
                if len(random_extents[tuple(x)]) > 0:
                    random_extents_set.add(tuple(x))
            return random_extents, random_extents_set
        else:
            return random_extents
