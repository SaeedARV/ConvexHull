# ConvexHull
Fast and scalable randomized approximation algorithms for convex hulls, plus a
deep learning baseline.

## Available solvers
- `ConvexHullviaExtents` — deterministic sampling of directional extents (supports any dimension).
- `ConvexHullviaMVEE` — MVEE-based approximation.
- `ConvexHullviaDeepHull` — DeepHull with selectable backends: the original ICNN
  or a smooth convex Lipschitz ICNN.

The DeepHull solver trains a convex function through an adversarial
classification loss, producing a tight convex hull approximation directly from
the supplied points. Install PyTorch to enable this method; the implementation
automatically falls back to CPU when GPU acceleration is not available. Choose
between the baseline ICNN (`method="original"`) and the learning-theoretic
convex variant (`method="convex"`) via the `method` argument.

```python
import numpy as np
from ConvexHullviaDeepHull import ConvexHullviaDeepHull

points = np.random.rand(30, 3)
deephull = ConvexHullviaDeepHull(max_epochs=150, method="convex", lipschitz_constant=1.0)
vertex_indices = deephull.predict_indices(points)
approx_points = points[vertex_indices]
```

## Benchmarking the methods

Use `benchmark_convex_hulls.py` to generate synthetic datasets (uniform, rotated
Gaussian, and annulus shells), run all solvers, and report accuracy/runtime
metrics against SciPy’s exact hull:

```bash
python3 benchmark_convex_hulls.py --datasets uniform,gaussian,annulus \
    --samples 200 --min-points 50 --max-points 50 --dimension 5
```

Key columns in the output table:

- `Vertex F1` / `Precision` / `Recall`: agreement with the exact hull vertices.
- `Area ratio`: predicted hull area divided by the exact hull area.
- `Coverage`: fraction of all input points lying inside the predicted hull.
- `Vertices`: average number of vertices returned by the solver.
- `Runtime (ms)`: wall-clock time per instance.

DeepHull results appear automatically when PyTorch is installed. Additional
training controls are exposed via command-line flags such as
`--deephull-epochs`, `--deephull-lambda`, `--deephull-epsilon`,
`--deephull-method`, `--deephull-lipschitz`, and `--dimension`
for higher-dimensional benchmarks. Use `--deephull-method both` (or a
comma-separated list like `original,convex`) to benchmark both DeepHull variants in a
single run.
