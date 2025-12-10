import numpy as np
from common import normalize

# ---------------- Backend detection ---------------- #

_HAS_CUPY = False
_HAS_GPU = False
cp = None

try:
    import cupy as _cp  # type: ignore

    if _cp.cuda.runtime.getDeviceCount() > 0:
        cp = _cp
        _HAS_CUPY = True
        _HAS_GPU = True
except Exception:
    cp = None
    _HAS_CUPY = False
    _HAS_GPU = False

_HAS_NUMBA = False
try:
    from numba import njit, prange  # type: ignore

    _HAS_NUMBA = True
except Exception:
    njit = None  # type: ignore
    prange = range  # type: ignore
    _HAS_NUMBA = False


# ---------------- Numba kernels (CPU, JIT) ---------------- #

if _HAS_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _numba_dot_matrix(X, D):
        """
        Compute X @ D.T where
          X: (n, d), D: (t, d)
        Returns: (n, t)
        """
        n, d = X.shape
        t = D.shape[0]
        out = np.empty((n, t), dtype=np.float64)
        for i in prange(n):
            for j in range(t):
                s = 0.0
                for k in range(d):
                    s += X[i, k] * D[j, k]
                out[i, j] = s
        return out

else:
    _numba_dot_matrix = None  # type: ignore


class ConvexHullviaExtents:
    """
    Class to compute and store the convex hull extents.

    This implementation supports three backends:
      - 'gpu'   : CuPy + CUDA (if available)
      - 'numba' : Numba-accelerated CPU
      - 'numpy' : Plain NumPy

    The default mode 'auto' will pick GPU if available, otherwise
    Numba if available, otherwise NumPy.
    """

    def __init__(self, points):
        self.points = np.array(points, dtype=np.float64)
        self.X = np.array(points, dtype=np.float64)
        self.extents, self.random_extents, self.extents_max = None, None, None

    # ---------- Backend selection helpers ---------- #

    @staticmethod
    def _select_backend(method: str):
        """
        Decide which backend to use.

        method in {'auto', 'gpu', 'numba', 'numpy'}.
        """
        method = method.lower()
        if method not in {"auto", "gpu", "numba", "numpy"}:
            raise ValueError(
                f"Unknown method '{method}' (expected 'auto', 'gpu', 'numba', 'numpy')."
            )

        if method == "gpu":
            if _HAS_GPU:
                return "gpu"
            # If requested GPU but not available, fall back to best CPU option.
            if _HAS_NUMBA:
                return "numba"
            return "numpy"

        if method == "numba":
            if _HAS_NUMBA:
                return "numba"
            return "numpy"

        if method == "numpy":
            return "numpy"

        # method == "auto"
        if _HAS_GPU:
            return "gpu"
        if _HAS_NUMBA:
            return "numba"
        return "numpy"

    @staticmethod
    def _compute_directions(d, t):
        """
        Generate t unit directions in R^d.
        For d == 2: uniform angle sampling.
        For d  >  2: normal sampling + normalization.
        """
        if d == 2:
            thetas = np.linspace(0.0, 2.0 * np.pi, t, endpoint=False)
            directions = np.stack((np.cos(thetas), np.sin(thetas)), axis=1)
        else:
            directions = np.random.normal(size=(t, d))
            norms = np.linalg.norm(directions, axis=1)
            norms[norms == 0.0] = 1.0
            directions /= norms[:, np.newaxis]
        return directions

    # ---------- Core linear algebra ---------- #

    def _dot_products_numpy(self, directions):
        """
        Return X @ directions.T using NumPy.
        Shapes:
          X: (n, d)
          directions: (t, d)
          result: (n, t)
        """
        return self.X @ directions.T

    def _dot_products_gpu(self, directions):
        """
        Return X @ directions.T using CuPy on GPU.
        Shapes as above.
        """
        if not _HAS_GPU:
            raise RuntimeError("GPU backend requested but no CUDA GPU is available.")

        # Move data to GPU
        X_gpu = cp.asarray(self.X)  # type: ignore
        D_gpu = cp.asarray(directions)  # type: ignore

        dp_gpu = X_gpu @ D_gpu.T  # (n, t) on GPU
        return dp_gpu

    def _dot_products_numba(self, directions):
        """
        Return X @ directions.T using a Numba-accelerated kernel.
        """
        if not _HAS_NUMBA or _numba_dot_matrix is None:
            # Graceful fallback
            return self._dot_products_numpy(directions)
        return _numba_dot_matrix(self.X, directions)

    # ---------- Public API ---------- #

    def get_extents(self, t=1000, return_max_extent=False, method: str = "auto"):
        """
        Calculates the directions of the convex hull's extents in any dimension.

        Parameters
        ----------
        t : int, optional
            Number of directions to sample. Default is 1000.
        return_max_extent : bool, optional
            If True, also returns the maximum extent direction for each point.
        method : {'auto', 'gpu', 'numba', 'numpy'}
            Backend to use. 'auto' prefers GPU if available.

        Returns
        -------
        extents : dict
            Dictionary mapping each point (as tuple) to a list of directions
            (NumPy arrays) where it is the supporting point.
        extents_max : dict, optional
            Dictionary mapping each point to its maximum extent direction.
            Only returned if return_max_extent is True.
        """
        n, d = self.X.shape
        backend = self._select_backend(method)

        directions = self._compute_directions(d, t)

        # Compute dot products according to backend
        if backend == "gpu":
            dp_gpu = self._dot_products_gpu(directions)
            # For each direction (axis=0 over columns), which point is maximal?
            # dp_gpu: (n, t)
            idx_support = cp.argmax(dp_gpu, axis=0)  # shape (t,)
            idx_support_host = cp.asnumpy(idx_support)

            extents = {tuple(x): [] for x in self.X}
            for i_dir in range(t):
                idx = int(idx_support_host[i_dir])
                x = tuple(self.X[idx])
                extents[x].append(directions[i_dir])

            if return_max_extent:
                # For each point (axis=1 over rows), which direction is maximal?
                idx_dir = cp.argmax(dp_gpu, axis=1)  # shape (n,)
                idx_dir_host = cp.asnumpy(idx_dir)

                extents_max = {}
                for i_point in range(n):
                    x = tuple(self.X[i_point])
                    dir_idx = int(idx_dir_host[i_point])
                    extents_max[x] = normalize(directions[dir_idx])
                return extents, extents_max

            return extents

        elif backend == "numba":
            dot_products = self._dot_products_numba(directions)
        else:  # 'numpy'
            dot_products = self._dot_products_numpy(directions)

        # CPU post-processing (same for NumPy / Numba backends)
        extents = {tuple(x): [] for x in self.X}

        # For each direction, find supporting point
        # dot_products: (n, t)
        max_idx_per_dir = np.argmax(dot_products, axis=0)
        for i_dir in range(t):
            idx = int(max_idx_per_dir[i_dir])
            x = tuple(self.X[idx])
            extents[x].append(directions[i_dir])

        if return_max_extent:
            # For each point, find direction with max extent
            max_idx_per_point = np.argmax(dot_products, axis=1)
            extents_max = {}
            for i_point in range(n):
                x = tuple(self.X[i_point])
                dir_idx = int(max_idx_per_point[i_point])
                extents_max[x] = normalize(directions[dir_idx])
            return extents, extents_max

        return extents

    def get_random_extents(self, t, return_approx_hull=False, method: str = "auto"):
        """
        Calculates randomized directional extents of the convex hull in any dimension.

        Parameters
        ----------
        t : int
            Number of random directions to sample.
        return_approx_hull : bool, optional
            If True, also returns the set of points that have at least one extent.
        method : {'auto', 'gpu', 'numba', 'numpy'}
            Backend to use. 'auto' prefers GPU if available.

        Returns
        -------
        random_extents : dict
            Dictionary mapping each point (as tuple) to a list of random directions
            where it is the supporting point.
        random_extents_set : set, optional
            Set of points that have at least one extent direction.
            Only returned if return_approx_hull is True.
        """
        n, d = self.X.shape
        backend = self._select_backend(method)

        # Random directions (unit vectors)
        random_directions = np.random.normal(size=(t, d))
        norms = np.linalg.norm(random_directions, axis=1)
        norms[norms == 0.0] = 1.0
        random_directions /= norms[:, np.newaxis]

        if backend == "gpu":
            dp_gpu = self._dot_products_gpu(random_directions)
            idx_support = cp.argmax(dp_gpu, axis=0)
            idx_support_host = cp.asnumpy(idx_support)

            random_extents = {tuple(x): [] for x in self.X}
            for i_dir in range(t):
                idx = int(idx_support_host[i_dir])
                x = tuple(self.X[idx])
                random_extents[x].append(random_directions[i_dir])

        elif backend == "numba":
            random_dot_products = self._dot_products_numba(random_directions)
            random_extents = {tuple(x): [] for x in self.X}
            max_idx_per_dir = np.argmax(random_dot_products, axis=0)
            for i_dir in range(t):
                idx = int(max_idx_per_dir[i_dir])
                x = tuple(self.X[idx])
                random_extents[x].append(random_directions[i_dir])

        else:  # 'numpy'
            random_extents = {tuple(x): [] for x in self.X}
            random_dot_products = self._dot_products_numpy(random_directions)
            max_idx_per_dir = np.argmax(random_dot_products, axis=0)
            for i_dir in range(t):
                idx = int(max_idx_per_dir[i_dir])
                x = tuple(self.X[idx])
                random_extents[x].append(random_directions[i_dir])

        if return_approx_hull:
            random_extents_set = {
                x for x, dirs in random_extents.items() if len(dirs) > 0
            }
            return random_extents, random_extents_set
        else:
            return random_extents

    # ---------- Master compute wrapper ---------- #

    def compute(
        self,
        t=1000,
        mode: str = "deterministic",
        method: str = "auto",
        return_max_extent: bool = False,
        return_approx_hull: bool = False,
    ):
        """
        Master wrapper to compute convex hull extents.

        Parameters
        ----------
        t : int
            Number of directions to sample.
        mode : {'deterministic', 'random'}
            - 'deterministic': uses a deterministic grid of directions on the sphere
              (calls get_extents).
            - 'random'      : uses random directions (calls get_random_extents).
        method : {'auto', 'gpu', 'numba', 'numpy'}
            Backend selector.
        return_max_extent : bool
            Forwarded to get_extents when mode='deterministic'.
        return_approx_hull : bool
            Forwarded to get_random_extents when mode='random'.

        Returns
        -------
        Depends on mode:
        - mode == 'deterministic':
            same as get_extents(...)
        - mode == 'random':
            same as get_random_extents(...)
        """
        mode = mode.lower()
        if mode not in {"deterministic", "random"}:
            raise ValueError("mode must be 'deterministic' or 'random'.")

        if mode == "deterministic":
            return self.get_extents(
                t=t, return_max_extent=return_max_extent, method=method
            )
        else:
            return self.get_random_extents(
                t=t, return_approx_hull=return_approx_hull, method=method
            )


### How to use it

# * Default (auto-select backend, deterministic directions):
# hull = ConvexHullviaExtents(points)
# extents = hull.compute(t=5000)  # uses GPU if available, else numba, else numpy

# * Force GPU (with graceful fallback to CPU if no GPU):
# extents = hull.compute(t=5000, method="gpu")

# * Force Numba or plain NumPy:
# extents = hull.compute(t=5000, method="numba")
# extents = hull.compute(t=5000, method="numpy")

# * Random directions + approximate hull set:
# random_extents, approx_hull_pts = hull.compute(
#     t=5000,
#     mode="random",
#     method="auto",
#     return_approx_hull=True,
# )
