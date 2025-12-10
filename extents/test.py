import numpy as np
from convexhull import ConvexHullviaExtents


def compare_extents_max(ext_a, ext_b, atol=1e-8):
    """
    Compare extents_max dicts: same keys, directions close.
    ext_* : dict mapping point (tuple) -> np.array(direction)
    """
    keys_a = set(ext_a.keys())
    keys_b = set(ext_b.keys())
    if keys_a != keys_b:
        return False

    for k in keys_a:
        v_a = np.asarray(ext_a[k])
        v_b = np.asarray(ext_b[k])
        if not np.allclose(v_a, v_b, atol=atol):
            return False
    return True


def summarize_extents(extents):
    """
    Simple summary: how many points are in the approx hull, total directions, etc.
    """
    num_support_points = sum(1 for dirs in extents.values() if len(dirs) > 0)
    total_dirs = sum(len(dirs) for dirs in extents.values())
    return num_support_points, total_dirs


def main():
    # Problem size
    n_points = 500
    dim = 3
    t = 400  # number of sampled directions

    # Generate random point cloud
    np.random.seed(41)
    points = np.random.randn(n_points, dim)

    hull = ConvexHullviaExtents(points)

    print("=== Deterministic extents: backend comparisons ===")
    # Baseline: pure NumPy
    np.random.seed(0)
    ext_numpy, extmax_numpy = hull.compute(
        t=t,
        mode="deterministic",
        method="numpy",
        return_max_extent=True,
    )
    base_support, base_total = summarize_extents(ext_numpy)
    print(f"[numpy] support points = {base_support}, total directions = {base_total}")

    # Auto backend vs numpy (with same seed)
    np.random.seed(0)
    ext_auto, extmax_auto = hull.compute(
        t=t,
        mode="deterministic",
        method="auto",
        return_max_extent=True,
    )
    auto_support, auto_total = summarize_extents(ext_auto)
    print(f"[auto ] support points = {auto_support}, total directions = {auto_total}")

    # Optional: explicit numba backend (falls back to numpy if numba not available)
    np.random.seed(0)
    ext_numba, extmax_numba = hull.compute(
        t=t,
        mode="deterministic",
        method="numba",
        return_max_extent=True,
    )
    numba_support, numba_total = summarize_extents(ext_numba)
    print(f"[numba] support points = {numba_support}, total directions = {numba_total}")

    # Optional: explicit gpu backend (falls back internally if no GPU)
    np.random.seed(0)
    ext_gpu, extmax_gpu = hull.compute(
        t=t,
        mode="deterministic",
        method="gpu",
        return_max_extent=True,
    )
    gpu_support, gpu_total = summarize_extents(ext_gpu)
    print(f"[gpu  ] support points = {gpu_support}, total directions = {gpu_total}")

    # Sanity checks: auto vs numpy should be numerically consistent
    # (we used the same seed => same sampled directions)
    assert compare_extents_max(
        extmax_numpy, extmax_auto
    ), "extents_max mismatch between numpy and auto backends"

    # Random mode tests
    print("\n=== Random extents: backend summaries ===")
    for method in ["numpy", "numba", "gpu", "auto"]:
        np.random.seed(123)  # same seed for all backends
        random_extents, approx_hull = hull.compute(
            t=t,
            mode="random",
            method=method,
            return_approx_hull=True,
        )
        num_support_points = len(approx_hull)
        total_dirs = sum(len(dirs) for dirs in random_extents.values())
        print(
            f"[{method:5}] approx hull size = {num_support_points}, "
            f"total directions = {total_dirs}"
        )

    print("\nAll tests completed without assertion failures.")


if __name__ == "__main__":
    main()
