import numpy as np
import matplotlib.pyplot as plt
from convex_hull import ConvexHullviaExtents, ConvexHullviaMVEE
from convex_hull.geometry import Ellipsoid

np.random.seed(41)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


class plots:
    """
    Class to handle plotting of convex hulls and extents.
    """

    def __init__(self, points, hull):
        self.points = np.array(points)
        self.hull = hull
        self.hull_points = points[hull.vertices]
        # Plot the points
        plt.plot(self.points[:, 0], self.points[:, 1], "o", label="Points")
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        self.plt = plt

    def convex_hull(self):
        """
        Plots the convex hull of a set of points in 2D.

        Parameters:
        points (array-like): An array of shape (n, 2) where n is the number of points.
        """
        points, hull, plt = self.points, self.hull, self.plt

        # Convert points to a numpy array
        points = np.array(points)

        # Plot the convex hull simplices
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], "k-")
        plt.scatter(0, 0, c="black", zorder=11111, label="Origin")

        plt.plot(
            points[hull.vertices[:], 0],
            points[hull.vertices[:], 1],
            "o",
            c="r",
            label="Hull Vertices",
        )
        plt.rcParams["figure.figsize"] = (10, 6)

        plt.fill(
            points[hull.vertices, 0],
            points[hull.vertices, 1],
            alpha=0.3,
            label="Convex Hull",
        )
        return plt

    def plot_directional_extents(self, t=1000, return_max_extent=False):
        """
        Plots the convex hull and its extents.

        Parameters:
        points (array-like): An array of shape (n, 2) where n is the number of points.
        """
        plt = self.plt
        points, plt = self.points, self.plt

        ConvHull = ConvexHullviaExtents(points)
        if return_max_extent:
            extents, extents_max = ConvHull.get_extents(t, True)
            for point, direction in extents_max.items():
                plt.arrow(
                    point[0],
                    point[1],
                    direction[0],
                    direction[1],
                    head_width=0.1,
                    head_length=0.2,
                    fc="m",
                    ec="m",
                    alpha=0.1,
                    # label="Max Extent",
                )
        else:
            extents = ConvHull.get_extents(t)

        for point, directions in extents.items():
            for direction in directions:
                plt.arrow(
                    point[0],
                    point[1],
                    1.5 * direction[0],
                    1.5 * direction[1],
                    head_width=0,
                    head_length=0,
                    fc="b",
                    ec="b",
                    alpha=0.02,
                    # label="Extents Cones",
                )
        return plt

    def plot_convex_hull_extents(self, t=50):
        """
        Plots the convex hull and its extents.

        Parameters:
        points (array-like): An array of shape (n, 2) where n is the number of points.
        """
        plt = self.plt
        points, plt = self.points, self.plt

        ConvHull = ConvexHullviaExtents(points)
        random_extents, random_extents_set = ConvHull.get_random_extents(t, True)
        for point, directions in random_extents.items():
            for direction in directions:
                plt.arrow(
                    point[0],
                    point[1],
                    direction[0],
                    direction[1],
                    head_width=0.1,
                    head_length=0.2,
                    fc="r",
                    ec="r",
                    alpha=0.2,
                    # label="Directional Extents",
                )
        return plt

    def AllviaExtents(self):
        plt = self.plt
        plt = self.convex_hull()
        plt = self.plot_directional_extents()
        plt = self.plot_convex_hull_extents()
        plt.legend(bbox_to_anchor=(1.01, 1.02))
        plt.show()
        return plt

    def plot_MVEE(self):
        """
        Plots the Minimum Volume Enclosing Ellipsoid (MVEE) in 2D.

        Parameters:
        c (array-like): Center of the MVEE.
        A (array-like): Shape matrix of the MVEE.
        """
        plt = self.plt
        E = Ellipsoid(self.points, method="Khachiyan")
        cs, Ps = E.center, E.shape_matrix
        print(cs, Ps)
        # Plot the ellipsoid
        plt.scatter(cs[0], cs[1], c="g", marker="x", label="MVEE center")
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = np.stack([np.cos(theta), np.sin(theta)])
        # Transform the circle to the ellipsoid
        L = np.linalg.cholesky(np.linalg.inv(Ps))  # A = (L^T L)^(-1)
        ellip = (L @ circle).T + cs
        plt.plot(ellip[:, 0], ellip[:, 1], "b-", label="MVEE")
        return plt

    def plot_MVEE_convex_hull_extents(self, t=5, kappa=165):
        """
        Plots the convex hull and its extents.

        Parameters:
        points (array-like): An array of shape (n, 2) where n is the number of points.
        """
        points, plt = self.points, self.plt

        ConvHull = ConvexHullviaMVEE(points)
        S, extents, U, rotated_vecs, perp_vecs = ConvHull.compute(
            return_extents=True, m=8, kappa=45
        )
        # S, extents, U, rotated_vecs, perp_vecs = ConvHull.compute(return_extents=True)
        for u in U:
            plt.arrow(
                0,
                0,
                u[0],
                u[1],
                head_width=0.1,
                head_length=0.2,
                fc="grey",
                ec="grey",
                alpha=0.9,
                zorder=1111111,
                # label="Presampled vectors",
            )

        for point, directions in extents.items():
            for direction in directions:
                plt.arrow(
                    point[0],
                    point[1],
                    direction[0],
                    direction[1],
                    head_width=0.1,
                    head_length=0.2,
                    fc="g",
                    ec="g",
                    alpha=0.2,
                    # label="Directional Extents",
                )

        # for point, directions in rotated_vecs.items():
        #     if point not in self.hull_points:
        #         continue
        #     for direction in directions:
        #         plt.arrow(
        #             point[0],
        #             point[1],
        #             direction[0],
        #             direction[1],
        #             head_width=0.1,
        #             head_length=0.2,
        #             fc="r",
        #             ec="r",
        #             alpha=0.2,
        #             # label="Directional Extents",
        #         )
        for point, direction in perp_vecs.items():
            if point not in self.hull_points:
                continue
            plt.arrow(
                point[0],
                point[1],
                direction[0],
                direction[1],
                head_width=0.1,
                head_length=0.5,
                fc="r",
                ec="r",
                alpha=0.4,
                # label="Directional Extents",
            )
        return plt

    def plot_MVEE_projections(self):
        points, plt = self.points, self.plt
        E = Ellipsoid(points)
        projected_points = np.array([E.project(p) for p in points])
        plt.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            c="orange",
            label="Projected Points",
        )
        for i, p in enumerate(points):
            plt.arrow(
                p[0],
                p[1],
                projected_points[i, 0] - p[0],
                projected_points[i, 1] - p[1],
                fc="orange",
                ec="orange",
                alpha=0.1,
            )
            v = E.normal_vector(projected_points[i])
            plt.arrow(
                projected_points[i, 0],
                projected_points[i, 1],
                v[0],
                v[1],
                fc="orange",
                ec="orange",
                head_width=0.1,
                head_length=0.5,
                alpha=0.1,
            )

        return plt

    def AllviaMVEE(self, return_max_extent=False):
        plt = self.plt
        plt = self.convex_hull()
        plt = self.plot_directional_extents(return_max_extent=return_max_extent)
        plt = self.plot_MVEE()
        plt = self.plot_MVEE_convex_hull_extents()
        if return_max_extent:
            plt = self.plot_MVEE_projections()
        plt.legend(bbox_to_anchor=(1.01, 1.02))
        plt.show()
        return plt

    def Plain(self):
        plt = self.plt
        plt = self.convex_hull()
        plt = self.plot_directional_extents()
        plt.legend(bbox_to_anchor=(1.01, 1.02))
        plt.show()
        return plt
